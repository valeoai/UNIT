import MinkowskiEngine as ME
import numpy as np
import torch


class VoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
            "unsupervised_instance_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries

    def __call__(self, batch):
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            num_queries=self.num_queries,
        )


class VoxelizeCollateFixed:
    def __init__(
        self,
        voxel_size=1,
        mode="test",
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
            "unsupervised_instance_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.mode = mode
        self.batch_instance = batch_instance
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

    def __call__(self, batch):
        return voxelize_fixed(
            batch,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
        )


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch


def voxelize(
    batch,
    ignore_label,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
):
    return_list = []
    for frame_id in range(len(batch[0])):
        (
            coordinates,
            features,
            labels,
            original_labels,
            filenames,
            inverse_maps,
            original_coordinates,
            idx,
        ) = ([], [], [], [], [], [], [], [])


        for sample in batch:
            idx.append(sample[frame_id][5])
            original_coordinates.append(sample[frame_id][4])
            original_labels.append(sample[frame_id][2])
            filenames.append(sample[frame_id][3])

            coords = np.floor(sample[frame_id][0] / voxel_size)

            # maybe this change (_, _, ...) is not necessary and we can directly get out
            # the sample coordinates?
            sample_coordinates, sample_features, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=torch.from_numpy(coords),
                features=sample[frame_id][1],
                # ignore_label=ignore_label,
                return_index=True,
                return_inverse=True,
            )
            inverse_maps.append(inverse_map)

            coordinates.append(sample_coordinates)
            features.append(torch.from_numpy(sample_features).float())
            if len(sample[frame_id][2]) > 0:
                sample_labels = sample[frame_id][2][unique_map]
                labels.append(torch.from_numpy(sample_labels).long())

        # Concatenate all lists
        input_dict = {"coords": coordinates, "feats": features}
        if len(labels) > 0:
            input_dict["labels"] = labels
            coordinates, features, labels = ME.utils.sparse_collate(coords=coordinates, feats=features, labels=labels)
        else:
            coordinates, features = ME.utils.sparse_collate(coords=coordinates, feats=features)
            labels = torch.Tensor([])

        if probing:
            return_list.append((
                NoGpu(
                    coordinates,
                    features,
                    original_labels,
                    inverse_maps,
                ),
                labels,
            ))
        
        if "labels" in input_dict:
            ### This is useles and makes no sense ###
            for i in range(len(input_dict["labels"])):
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, 0],
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            #########################################

            list_labels = input_dict["labels"]

            target = []
            target_full = []

            if len(list_labels[0].shape) == 1:
                for batch_id in range(len(list_labels)):
                    label_ids = list_labels[batch_id].unique()
                    if 255 in label_ids:
                        label_ids = label_ids[:-1]

                    target.append(
                        {
                            "labels": label_ids,
                            "masks": list_labels[batch_id]
                            == label_ids.unsqueeze(1),
                        }
                    )
            else:
                if mode == "test":
                    for i in range(len(input_dict["labels"])):
                        target.append(
                            {"point2segment": input_dict["labels"][i][:, 0]}
                        )
                        target_full.append(
                            {
                                "point2segment": torch.from_numpy(
                                    original_labels[i][:, 0]
                                ).long()
                            }
                        )
                else:
                    target = get_instance_masks(
                        list_labels,
                        task=task,
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                    )
                    if "train" not in mode:
                        target_full = get_instance_masks(
                            [torch.from_numpy(l) for l in original_labels],
                            task=task,
                            ignore_class_threshold=ignore_class_threshold,
                            filter_out_classes=filter_out_classes,
                            label_offset=label_offset,
                        )
        else:
            target = []
            target_full = []
            coordinates = []
            features = []

        if "train" not in mode:
            return_list.append((
                NoGpu(
                    coordinates,
                    features,
                    original_labels,
                    inverse_maps,
                    target_full,
                    original_coordinates,
                    idx,
                ),
                target,
                filenames,
            ))
        else:
            return_list.append((
                NoGpu(
                    coordinates,
                    features,
                    original_labels,
                    inverse_maps,
                    original_coordinates=original_coordinates,
                ),
                target,
                filenames,
            ))
    return return_list


def voxelize_fixed(
    batch,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
):
    nogpu_list = []
    labels_list = []
    filenames_list = []
    original_labels_list = []
    for frame_id in range(len(batch[0])):
        (
            coordinates,
            features,
            labels,
            original_labels,
            filenames,
            inverse_maps,
            original_coordinates,
            idx,
        ) = ([], [], [], [], [], [], [], [])


        for sample in batch:
            idx.append(sample[frame_id][5])
            original_coordinates.append(sample[frame_id][4])
            original_labels.append(sample[frame_id][2])
            filenames.append(sample[frame_id][3])

            coords = np.floor(sample[frame_id][0] / voxel_size)

            # maybe this change (_, _, ...) is not necessary and we can directly get out
            # the sample coordinates?
            sample_coordinates, sample_features, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=torch.from_numpy(coords),
                features=sample[frame_id][1],
                return_index=True,
                return_inverse=True,
            )
            inverse_maps.append(inverse_map)

            coordinates.append(sample_coordinates)
            features.append(torch.from_numpy(sample_features).float())
            if len(sample[frame_id][2]) > 0:
                sample_labels = sample[frame_id][2][unique_map]
                labels.append(torch.from_numpy(sample_labels).long())

        # Concatenate all lists
        coordinates, features = ME.utils.sparse_collate(coords=coordinates, feats=features)

        nogpu_list.append(
            NoGpu(
                coordinates,
                features,
                original_labels=original_labels,
                inverse_maps=inverse_maps,
                original_coordinates=original_coordinates,
            )
        )
        labels_list.append(labels)
        filenames_list.append(filenames)
        original_labels_list.append(original_labels)

    return_list = []
    if "train" not in mode:
        for i in range(len(nogpu_list)):
            target_full = []
            for batch_id in range(len(original_labels_list[i])):
                instance_ids = np.unique(np.concatenate([original_labels_list[j][batch_id][:, 1] for j in range(len(original_labels_list))]))
                masks = torch.from_numpy(original_labels_list[i][batch_id][:, 1] == instance_ids[:, None])
                conditions = instance_ids == -1
                masks = masks[~conditions]
                target_full.append({"masks": masks})

            nogpu_list[i].target_full = target_full
    for i in range(len(nogpu_list)):
        # create target
        target = []
        for batch_id in range(len(labels_list[i])):  # TODO vectorialize
            instance_ids = torch.unique(torch.concatenate([labels_list[j][batch_id][:, 1] for j in range(len(labels_list))]))
            masks = labels_list[i][batch_id][:, 1] == instance_ids[:, None]
            conditions = instance_ids == -1
            masks = masks[~conditions]
            target.append({"masks": masks})

        return_list.append((nogpu_list[i], target, filenames_list[i]))
    return return_list


def get_instance_masks(
    list_labels,
    task,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
):
    target = []

    for batch_id in range(len(list_labels)):  # TODO vectorialize
        instance_ids, indexes, counts = np.unique(list_labels[batch_id][:, 1], return_index=True, return_counts=True)
        label_ids = list_labels[batch_id][indexes][:, 0]
        masks = list_labels[batch_id][:, 1] == torch.from_numpy(instance_ids)[:, None]
        conditions = np.logical_or(np.isin(label_ids, filter_out_classes), instance_ids == -1)
        label_ids = label_ids[~conditions]
        masks = masks[~conditions]

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            for label_id in label_ids.unique():
                masking = label_ids == label_id

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            target.append({"labels": label_ids, "masks": masks})
        elif task == "unsupervised_instance_segmentation":
            target.append({"masks": masks})
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            target.append({"labels": l, "masks": masks})
    return target


def get_instance_masks_fixed(
    list_labels,
):
    target = []

    for batch_id in range(len(list_labels)):  # TODO vectorialize
        instance_ids = np.unique(list_labels[batch_id][:, 1])
        masks = list_labels[batch_id][:, 1] == torch.from_numpy(instance_ids)[:, None]
        conditions = instance_ids == -1
        masks = masks[~conditions]
        target.append({"masks": masks})

    return target

class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        target_full=None,
        original_coordinates=None,
        idx=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.target_full = target_full
        self.original_coordinates = original_coordinates
        self.idx = idx


class NoGpuMask:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        masks=None,
        labels=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels
