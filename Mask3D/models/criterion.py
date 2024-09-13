# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch_scatter



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
        pad_targets=False,
    ):
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.pad_targets = pad_targets

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_masks(self, outputs, targets, indices, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Pad the targets with zeros to force assignment for all queries
        if self.pad_targets:
            targets = [{'masks': F.pad(t['masks'], (0, 0, 0, max(outputs["pred_masks"][0].shape[1] - t['masks'].shape[0], 0)))} for t in targets]

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.loss_masks(outputs, targets, indices, mask_type)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.loss_masks(aux_outputs, targets, indices, mask_type)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class Consistency(nn.Module):
    """This class computes the consistency loss using an average scatter opetation.
    """

    def __init__(
        self,
        num_points=-1,
        compute_aux_loss=True,
        cost_consistency=1.,
    ):
        super().__init__()
        self.compute_aux_loss = compute_aux_loss

        # pointwise mask loss parameters
        self.num_points = num_points
        self.cost_consistency = cost_consistency

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
            outputs: List of List of tensors, such that len(outputs) == batch_size.
            targets: list of List of tensors, such that len(targets) == batch_size.
        """
        outputs = [[torch.stack([outputs[fid]['pred_masks'][bid],] + [outputs[fid]['aux_outputs'][aid]['pred_masks'][bid] 
                                                          for aid in range(len(outputs[fid]['aux_outputs']))])
                    for bid in range(len(outputs[fid]['pred_masks']))]
                    for fid in range(len(outputs))]
        n_objects = 0
        if self.num_points != -1:
            raise NotImplementedError("num_points not implemented for consistency loss")
        loss = torch.zeros(outputs[0][0].shape[0], dtype=torch.float32, device=targets[0][0]['masks'].device)
        for bid in range(len(targets[0])): # batch index
            indexes1 = targets[0][bid]['masks'].to(torch.int32).argmax(0)
            indexes2 = targets[-1][bid]['masks'].to(torch.int32).argmax(0)
            fmap1 = torch_scatter.scatter(outputs[0][bid].detach(), indexes1, dim=1, reduce='mean', dim_size=len(targets[0][bid]['masks']))
            fmap2 = torch_scatter.scatter(outputs[-1][bid], indexes2, dim=1, reduce='mean', dim_size=len(targets[0][bid]['masks']))
            # Ignore objects that are not seen in one of the frames
            mask_object = torch.logical_and(fmap1[0].sum(1) != 0, fmap2[0].sum(1) != 0)
            CE = F.cross_entropy(fmap2.transpose(1,2), F.softmax(fmap1, dim=2).transpose(1,2), reduction='none')
            loss += CE[:, mask_object].mean(1)
            n_objects += mask_object.sum()
        if n_objects == 0:
            return {"loss_consistency": loss[0]}
        else:
            output = {"loss_consistency": loss[0] / n_objects}
            output.update({f"loss_consistency_{i}": loss[i + 1] / n_objects for i in range(len(loss) - 1)})
            return output