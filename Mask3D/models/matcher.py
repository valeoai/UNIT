# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np
from lapjv import lapjv


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def unsupervised_forward(self, pred_masks, targets, mask_type):
        """More memory-friendly matching"""
        bs = len(targets)
        num_queries = pred_masks[0].shape[1]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_mask = pred_masks[b].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target

            tgt_mask = targets[b][mask_type]

            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            if self.num_points != -1:
                point_idx = torch.randperm(
                    tgt_mask.shape[1], device=tgt_mask.device
                )[: int(self.num_points * tgt_mask.shape[1])]
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )
            else:
                # cost_mask = batch_sigmoid_ce_loss_jit(
                cost_mask = batch_sigmoid_ce_loss(
                    out_mask, tgt_mask
                )

                # Compute the dice loss betwen masks
                # cost_dice = batch_dice_loss_jit(
                cost_dice = batch_dice_loss(
                    out_mask, tgt_mask
                )
                

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            # indices.append(linear_sum_assignment(C))
            indices.append(LSA(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        return self.unsupervised_forward([pred_mask.detach().to(torch.float32) for pred_mask in outputs['pred_masks']], targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

def LSA(cost_matrix):
    i, j = cost_matrix.shape
    if i == j:
        row_ind, col_ind, _ = lapjv(cost_matrix)
        return np.arange(i), row_ind
    elif i > j:
        cost_padded = np.concatenate([cost_matrix, np.full((i, i - j), 100.)], axis=1)
        row_ind, col_ind, _ = lapjv(cost_padded)
        return col_ind[:j], np.arange(j)
    else:
        cost_padded = np.concatenate([cost_matrix, np.full((j - i, j), 100.)], axis=0)
        row_ind, col_ind, _ = lapjv(cost_padded)
        return np.arange(i), row_ind[:i]
