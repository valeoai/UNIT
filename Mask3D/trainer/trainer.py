from contextlib import nullcontext
import statistics
import os
import re
from benchmark.evaluate_semantic_instance import s_assoc_conf
from collections import defaultdict

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.criterion import Consistency


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        self.mask_type = "masks"

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)

        self.forward_queries = config.general.forward_queries
        self.queries_dropout = config.general.queries_dropout

        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_consistency": 1.0,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:  # to check
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = []

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        self.train_on_all_frames = config.general.train_on_all_frames
        
        if config.general.consistency_loss:
            self.consistency = True
            self.consistency_criterion = Consistency()
        else:
            self.consistency = False

        # misc
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        # try:
        #     self.test_dataset = hydra.utils.instantiate(
        #         self.config.data.test_dataset
        #     )
        # except:
        #     pass

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(
        self, x, queries=None, raw_coordinates=None, is_eval=False, queries_dropout=0.0
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                queries=queries,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                queries_dropout=queries_dropout,
            )
        return x

    def training_step(self, batch, batch_idx):
        queries = None
        outputs = []
        targets = []
        for frame_id in range(len(batch)):
            data, target, _ = batch[frame_id]

            raw_coordinates = None
            if self.config.data.add_raw_coordinates:
                raw_coordinates = data.features[:, -3:]
                data.features = data.features[:, :-3]

            data = ME.SparseTensor(
                coordinates=data.coordinates,
                features=data.features,
                device=self.device,
            )
            output = self.forward(
                data,
                queries=queries,
                raw_coordinates=raw_coordinates,
                queries_dropout=self.queries_dropout if frame_id == 0 else 0.0,
            )

            if self.train_on_all_frames or self.consistency:
                outputs.append(output)
                targets.append(target)

            if self.forward_queries:
                queries = output["queries"]
        
        if self.train_on_all_frames:
            output = self.fuse_outputs(outputs)
            target = self.fuse_targets(targets)

        losses = self.criterion(output, target, mask_type=self.mask_type)

        if self.consistency:
            del output, target
            torch.cuda.empty_cache()
            losses_consistency = self.consistency_criterion(outputs, targets)
            losses.update(losses_consistency)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        self.log_dict(logs, sync_dist=True)
        loss = sum(losses.values())
        self.training_step_outputs.append(loss.detach())
        self.log('loss', loss.detach(), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        train_loss = sum([out.cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results, sync_dist=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.test_epoch_end(outputs)
        self.validation_step_outputs.clear()

    def eval_step(self, batch, batch_idx):
        queries = None
        outputs = []
        targets = []
        inverse_maps = []
        targets_full = []
        file_names = []
        for frame_id in range(len(batch)):
            data, target, file_name = batch[frame_id]
            inverse_map = data.inverse_maps
            target_full = data.target_full

            raw_coordinates = None
            if self.config.data.add_raw_coordinates:
                raw_coordinates = data.features[:, -3:]
                data.features = data.features[:, :-3]

            data = ME.SparseTensor(
                coordinates=data.coordinates,
                features=data.features,
                device=self.device,
            )

            output = self.forward(
                data,
                queries=queries,
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
            outputs.append(output)
            targets.append(target)
            inverse_maps.append(inverse_map)
            targets_full.append(target_full)
            file_names.append(file_name)

            if self.forward_queries:
                queries = output["queries"]
        # Concatenate all lists (?)
        fused_outputs = self.fuse_outputs(outputs)
        targets = self.fuse_targets(targets)

        # Only first frame in the sequence will be evaluated for now, (to keep retro-compatibility)
        masks = [outputs[0]["pred_masks"][bid].detach().argmax(1).cpu()[inverse_maps[0][bid]] for bid in range(len(inverse_maps[0]))]
        names = file_names[0]
        self.preds.append([names, masks])

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            losses = self.criterion(
                fused_outputs, targets, mask_type=self.mask_type
            )

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)


        if self.config.data.test_mode != "test":
            pred = {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}
        else:
            pred =  0.0
        self.validation_step_outputs.append(pred)
        return pred

    def fuse_outputs(self, outputs):
        fused_outputs = {"pred_masks": [], "aux_outputs": []}
        # pred_masks
        for bid in range(len(outputs[0]["pred_masks"])):
            fused_outputs["pred_masks"].append(torch.cat([outputs[fid]["pred_masks"][bid] for fid in range(len(outputs))]))
        # aux_outputs
        for lid in range(len(outputs[0]["aux_outputs"])):
            ls = []
            for bid in range(len(outputs[0]["aux_outputs"][0]['pred_masks'])):
                ls.append(torch.cat([outputs[fid]["aux_outputs"][lid]['pred_masks'][bid] for fid in range(len(outputs))]))
            fused_outputs["aux_outputs"].append({'pred_masks':ls})
        return fused_outputs

    def fuse_targets(self, targets):
        fused_targets = []
        # masks
        for bid in range(len(targets[0])):
            pad_length = max([targets[fid][bid]['masks'].shape[0] for fid in range(len(targets))])
            fused_targets.append({'masks':torch.cat([torch.nn.functional.pad(targets[fid][bid]['masks'], (0,0,0,pad_length - targets[fid][bid]['masks'].shape[0])) for fid in range(len(targets))], 1)})
        return fused_targets

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(
        self, mask, inverse_map, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        return mask

    def eval_instance_epoch_end(self):
        # a = self.all_gather(self.preds) # doesn't work with dict
        if len(self.preds) > 10:
            if self.validation_dataset.dataset_name == "semantic_kitti":
                gt_data_path = self.validation_dataset.data_dir[0].parent / "labels"

                S_assoc = []
                for pred in self.preds:
                    for bid in range(len(pred[0])):
                        frame = int(pred[0][bid].split(".")[0].rsplit('/')[1])
                        labels = np.fromfile(os.path.join(gt_data_path, f"{frame:06d}" + ".label"), dtype=np.int32)
                        instance = labels >> 16
                        S = s_assoc_conf(pred[1][bid], instance)
                        S_assoc.append(S)
                S = [sum(S_assoc), len(S_assoc)]
                S = self.all_gather(S)
                self.log_dict({"val_s_assoc": sum(S[0])/ sum(S[1]).item()}, rank_zero_only=True)
            else:
                raise NotImplementedError
        self.preds.clear()
            
    def test_epoch_end(self, outputs):
        self.eval_instance_epoch_end()

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )

        self.log_dict(dd, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        assert len(batch) == 1, "batch size > 1 not implemented in inference"

        try:
            self.current_sequence
        except AttributeError:
            self.current_sequence = None

        data, _, file_names = batch[0]
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_coordinates = data.original_coordinates

        # logics if new sequence is detected
        if self.validation_dataset.dataset_name == "semantic_kitti":
            sequence = file_names[0].split('/', 1)[0]
        elif self.validation_dataset.dataset_name.startswith("pandaset"):
            sequence = file_names[0].split('/', 1)[0]
        elif self.validation_dataset.dataset_name == "nuscenes":
            sequence = int(file_names[0].split('/')[-1].split('.',1)[0].rsplit('__',1)[1])
            if self.current_sequence is not None:
                assert sequence - self.current_sequence < 10000000000000, "sequence reading went wrong"
                if sequence - self.current_sequence < 75000:
                    self.current_sequence = sequence
                else:
                    self.current_sequence = sequence
                    sequence = None
        elif self.validation_dataset.dataset_name == "nuscenes_keyframes":
            sequence = int(file_names[0].split('/')[-1].split('.',1)[0].rsplit('__',1)[1])
            if self.current_sequence is not None:
                assert sequence - self.current_sequence < 10000000000000, "sequence reading went wrong"
                if sequence - self.current_sequence < 525000:
                    self.current_sequence = sequence
                else:
                    self.current_sequence = sequence
                    sequence = None
        if sequence != self.current_sequence:
            self.current_sequence = sequence
            self.map_id = dict()
            self.id_centroid = np.full((self.config['model']['num_queries'],3), np.inf)
            self.ground = None
            self.queries = None
            self.current_mapping = 0
            self.staleness = np.zeros(self.config['model']['num_queries'], dtype=np.int32)

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        output = self.forward(
            data,
            queries=self.queries,
            raw_coordinates=raw_coordinates,
            is_eval=True,
        )
        if self.forward_queries:
            self.queries = output["queries"]

        asset_path = str(self.validation_dataset.data_dir[0]).replace("sequences", f"assets/{self.config.general.experiment_name}")[:-11]
        fname = os.path.join(asset_path, file_names[0].rsplit(".")[0] + ".seg")
        if self.validation_dataset.dataset_name == "semantic_kitti":
            asset_path = str(self.validation_dataset.data_dir[0]).replace("sequences", f"assets/{self.config.general.experiment_name}")[:-11]
            fname = os.path.join(asset_path, file_names[0].rsplit(".")[0] + ".seg")
        elif self.validation_dataset.dataset_name.startswith("pandaset"):
            asset_path = self.validation_dataset.data_dir +  f"/assets/{self.config.general.experiment_name}"
            fname = os.path.join(asset_path, sequence, 'lidar', file_names[0].rsplit('/')[-1].rsplit(".")[0] + ".seg")
        elif self.validation_dataset.dataset_name.startswith("nuscenes"):
            asset_path = self.validation_dataset.data_dir +  f"/assets/{self.config.general.experiment_name}"
            fname = os.path.join(asset_path, re.sub('/', '/LIDAR_TOP/', file_names[0])[:-4] + ".seg")
        else:
            raise NotImplementedError

        if not os.path.isdir(fname.rsplit('/', 1)[0]):
            os.makedirs(fname.rsplit('/', 1)[0])

        masks_affinity = output["pred_masks"][0].cpu()
        masks = masks_affinity.argmax(1)[inverse_maps[0]]
        masks = masks.numpy()

        self.staleness += 1

        # Find the ground id in the new masks. It's defined by proxy as the argmax
        if self.ground is None:
            unique, count = np.unique(masks, return_counts=True)
            ground_id = unique[count.argmax()]
            self.ground = ground_id
            self.map_id[ground_id] = 0

        if self.validation_dataset.dataset_name == "semantic_kitti":
            distance = 10
        elif self.validation_dataset.dataset_name.startswith("pandaset"):
            distance = 10
        elif self.validation_dataset.dataset_name == "nuscenes":
            distance = 5
        elif self.validation_dataset.dataset_name == "nuscenes_keyframes":
            distance = 20
        else:
            raise NotImplementedError
        
        for segment in np.unique(masks):
            if segment == self.ground:
                continue
            # find centroids
            centroid = np.mean(original_coordinates[0][masks == segment], axis=0)
            if np.linalg.norm(centroid - self.id_centroid[segment]) > distance:# or self.staleness[segment] > 10:
                self.current_mapping += 1
                self.map_id[segment] = self.current_mapping
            self.id_centroid[segment] = centroid
            self.staleness[segment] = 0

        for segment in np.unique(masks):
            if segment == self.ground:
                continue
            # find centroids
            centroid = np.mean(original_coordinates[0][masks == segment], axis=0)
            if np.linalg.norm(centroid - self.id_centroid[segment]) > 10:
                self.current_mapping += 1
                self.map_id[segment] = self.current_mapping
            self.id_centroid[segment] = centroid

        masks = np.vectorize(self.map_id.__getitem__)(masks).astype(np.int32)


        masks.tofile(fname)

        return
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
    
    def predict_dataloader(self):
        "Used for visualisation"
        dataset = hydra.utils.instantiate(
            self.config.data.predict_dataset
        )
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.predict_dataloader,
            dataset,
            collate_fn=c_fn,
        )
