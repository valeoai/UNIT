import os
import yaml
import numpy as np
from tqdm import tqdm

import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

# 
# -1 indicates points not part of any predicted object
#
NO_PREDICTION_ID = -1

#
# 0 indicates points not part of any GT object
#
NO_GROUND_TRUTH_ID = 0


def s_assoc_quantities(gt_id_point, pred_id_point, min_points, lut, frame_id):
    """
    Usable for either scanwise and temporal metrics.

    Compute and store all the quantities needed to compute s_assoc:
    - size of predicted segments
    - size of GT segments
    - size of intersection between GT and predicted segments
    - number of GT segments

    All results are stored in the lookup table `lut` passed as arguments

    After looping over all frames, s_assoc can computed using the function
    `s_assoc` below and which uses the information stored in the LUT

    stats_ids is a dictionnary to monitor usage of IDs over the frames and 
    facilitate debugging.

    Inputs:
    - gt_id_point: pointwise ground truth instance ID
    - pred_id_point: pointwise predicted instance ID
    - min_points: threshold on the size of the segments

    - stats_ids and frame_id are used solely to help detects bug
    """

    # Get list of IDs for ground truth and prediction
    list_gt_id = np.unique(gt_id_point)
    list_pred_id = np.unique(pred_id_point)
    if len(list_gt_id[list_gt_id != NO_GROUND_TRUTH_ID]) == 0:
        print(f"No GT annotation on frame {frame_id}")
        return lut

    pred_masks = []
    # --- Loop over prediction to compute size of predicted objects
    for pred_id in list_pred_id:
        
        # ID that indicates absence of prediction: go to next actual prediction
        if pred_id == NO_PREDICTION_ID:
            pred_masks.append(None)
            continue            
        
        # Binary mask for current prediction
        pred_id_mask = (pred_id_point == pred_id)
        pred_masks.append(pred_id_mask)
        
        # Accumulate to find size of predicted segment
        if lut["size_pred"].get(pred_id) is None:
            lut["size_pred"][pred_id] = pred_id_mask.sum()
        else:
            lut["size_pred"][pred_id] += pred_id_mask.sum()


    # --- Loop over GT to compute size of GT objects
    for gt_id in list_gt_id:

        # ID that indicates absence of GT annotation: skip
        if gt_id == NO_GROUND_TRUTH_ID: 
            continue

        # Mask for current GT object
        gt_id_mask = (gt_id_point == gt_id)

        # If GT object is too small, then pass (as in 4D-PLS implementation)
        # Note that for 4D-segments this condition remove temporal slices of
        # GT segments. Weird but kept to stick to behaviour of 4D-PLS implementation.
        if gt_id_mask.sum() <= min_points:
            continue

        # Accumulate to find size of predicted segment
        if lut["size_gt"].get(gt_id) is None:
            lut["size_gt"][gt_id] = gt_id_mask.sum()
            # ID was not in dictionnary -> new GT object is found
            lut["num_objects"] += 1
        else:
            lut["size_gt"][gt_id] += gt_id_mask.sum()

        # Loop over each prediction to find overlap with current GT object
        for pred_id, pred_id_mask in zip(list_pred_id, pred_masks):

            # ID that indicates absence of prediction: go to next actual prediction
            if pred_id == NO_PREDICTION_ID: continue
            
            # # Binary mask for current prediction
            # pred_id_mask = (pred_id_point == pred_id)
            
            # Intersection
            intersection = (pred_id_mask & gt_id_mask)

            # Intersection is empty, skip as in official definition of s_assoc
            if intersection.sum() == 0:
                continue

            # Store intersection for current pairs (gt_id, pred_id)
            key = (gt_id, pred_id)
            if lut["inter"].get(key) is None:
                lut["inter"][key] = intersection.sum()
            else:
                lut["inter"][key] += intersection.sum()

    return lut


def s_assoc(lut):
    """
    Compute S_assoc based on the information stored in lut.

    The scripts compute the non-normalized s_assoc `s_assoc_unorm`:

        sum_{t} [ |t|^{-1} sum_{s \cap t \neq 0} [ TPA(s, t) IoU(s, t) ] ]
    
        where `t` are the GT objects and `s` the predicted object

    and then return:
        s_assoc = s_assoc_unorm / num_objects
    """
    
    s_assoc_unorm = 0
    IoU_star = 0.

    # Loop over all GT objects "t" in the formula
    for gt_id in lut["size_gt"].keys():

        # Loop over each (gt_id, pred_id) pairs to compute the inner sum
        # "sum_{s \cap t \neq 0} [ TPA(s, t) IoU(s, t) ]"
        inner_sum = 0
        # Also gather the IoU*
        max_IoU = 0.
        for id_pairs in lut["inter"].keys():

            # Only consider pairs (gt_id, pred_id) involding the current GT ID
            if id_pairs[0] != gt_id: 
                continue
            
            # Extact TPA, aka size of intersection between GT mask and predicted mask
            TPA = lut["inter"][id_pairs]
            
            # Compute union betweenn GT mask and predicted mask
            union = lut["size_gt"][gt_id] + lut["size_pred"][id_pairs[1]] - TPA

            # Get IoU
            IoU = TPA / union
            max_IoU = max(max_IoU, IoU)

            # Acculumate for inner sum
            inner_sum += TPA * IoU
        
        # Divide `inner_sum` by size of object and accumulate
        s_assoc_unorm += inner_sum / lut["size_gt"][gt_id]

        IoU_star += max_IoU

    return s_assoc_unorm / lut["num_objects"], IoU_star / lut["num_objects"]


def compute_s_assoc_kitti(which_segment):

    # For class mapping
    with open("semantic-kitti.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml["learning_map"]
    learning_map = np.vectorize(learning_map.__getitem__)

    # Choose which implementation to use
    # Look up table for s_assoc computation
    lut_temporal_0 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_temporal_50 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_scanwise_0 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_scanwise_50 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}

    # Loop over all validation set 
    for idx_frame in tqdm(range(4071), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"):

        # Load GT label
        label_filename = f"/datasets_local/semantic_kitti/dataset/sequences/08/labels/{idx_frame:06d}.label"
        labels_inst = np.fromfile(label_filename, dtype=np.uint32)
        
        # Extract class label and apply official mapping
        labels_cls = labels_inst & 0xFFFF  # delete high 16 digits binary
        labels_cls = learning_map(labels_cls).astype(np.uint32)

        # Load prediction
        # NOTE: Make sure "NO_PREDICTION_ID" encodes points not part of any predicted object
        pred = f"../datasets/semantickitti/assets/{which_segment}/08/{idx_frame:06d}.seg"
        pred = np.fromfile(pred, dtype=np.int32)
        
        # As in semantic segmentation, 4D-PLS implementation filters out points marked 
        # with noisy / ignore semantic labels
        where_ignore = (labels_cls == 0)
        labels_inst = labels_inst[~where_ignore]
        pred = pred[~where_ignore]

        # Find points where there is no instance annotation and marked them with dedicated ID
        no_gt = ((labels_inst >> 16) == 0) # <- Use convention in semantic kitti labels
        labels_inst[no_gt] = NO_GROUND_TRUTH_ID
        
        # Compute scores for this frame
        # When computing scanwise metric, make sure IDs are not shared between frames
        if metric == "scanwise":
            where_no_prediction = (pred == NO_PREDICTION_ID)
            # Encode ID on 64 bits
            pred_scanwise = pred.astype(np.int64)
            labels_inst_scanwise = labels_inst.astype(np.int64)
            # First 32 bits encode frame index
            pred_scanwise = (idx_frame << 32) + pred_scanwise
            labels_inst_scanwise = (idx_frame << 32) + labels_inst_scanwise
            # Need to re-apply masks where there is no GT or no prediction
            labels_inst_scanwise[no_gt] = NO_GROUND_TRUTH_ID
            pred_scanwise[where_no_prediction] = NO_PREDICTION_ID
        # 
        lut_temporal_0 = s_assoc_quantities(labels_inst, pred, 0, lut_temporal_0, idx_frame)
        lut_temporal_50 = s_assoc_quantities(labels_inst, pred, 50, lut_temporal_50, idx_frame)
        lut_scanwise_0 = s_assoc_quantities(labels_inst_scanwise, pred_scanwise, 0, lut_scanwise_0, idx_frame)
        lut_scanwise_50 = s_assoc_quantities(labels_inst_scanwise, pred_scanwise, 50, lut_scanwise_50, idx_frame)

    # Final result
    print("Scores:")
    S_assoc, IoU_star = s_assoc(lut_temporal_0)
    print(
        f"Temporal S_assoc: {S_assoc:.5f}. Temporal IoU*: {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_temporal_50)
    print(
        f"Temporal S_assoc filtered (50 pts): {S_assoc:.5f}. Temporal IoU* filtered (50 pts): {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_scanwise_0)
    print(
        f"Scanwise S_assoc: {S_assoc:.5f}. Scanwise IoU*: {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_scanwise_0)
    print(
        f"Scanwise S_assoc filtered (50 pts): {S_assoc:.5f}. Scanwise IoU* filtered (50 pts): {IoU_star:.5f} "
    )


def compute_s_assoc_nuscenes(which_segment, nusc):
    phase_scenes = create_splits_scenes()["val"]

    # Extract class mapping
    learning_map = LidarsegClassMapper(nusc)
    assert learning_map.ignore_class['index'] == 0
    assert len(learning_map.coarse_name_2_coarse_idx_mapping) == 17

    # Look up table for s_assoc computation
    lut_temporal_0 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_temporal_50 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_scanwise_0 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}
    lut_scanwise_50 = {"inter": {}, "size_gt": {}, "size_pred": {}, "num_objects": 0}

    # Loop over all validation set
    frame_idx = 0 # Number of processed scans 
    window_idx, shift_id, max_id_seen = 0, 0, 0 # Window index for 4D-Seg
    for idx_seq, sequence in enumerate(tqdm(nusc.scene, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")):

        # Skip scenes not in validation set
        if sequence["name"] not in phase_scenes:
            continue
        
        # Loop through keyframes for current scene
        current_sample_token = sequence["first_sample_token"]

        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)

            # Load GT label
            label_filename = f"/datasets_master/nuscenes/panoptic/v1.0-trainval/{current_sample['data']['LIDAR_TOP']}_panoptic.npz"    
            labels_inst = np.load(label_filename)['data'].astype(np.int32)

            # Extract class label and apply official mapping
            labels_cls = (labels_inst // 1000).astype(np.int32)
            labels_cls = learning_map.convert_label(labels_cls).astype(np.int32)

            # Load prediction
            sample_data = nusc.get("sample_data", current_sample["data"]["LIDAR_TOP"])["filename"]
            windows = not os.path.exists(os.path.join(which_segment, sample_data[:-4] + ".seg"))
            if not windows:
                pred = os.path.join(which_segment, sample_data[:-4] + ".seg")
                pred = np.fromfile(pred, dtype=np.int32) # Saved in int 32
            else:
                while True:
                    pred = os.path.join(
                        which_segment,
                        sample_data[:-4] + f"_{window_idx}.seg"
                    )
                    if os.path.exists(pred):
                        break
                    else:
                        window_idx += 1
                        shift_id = max_id_seen
                pred = np.fromfile(pred, dtype=np.int16) 
                pred = pred.astype(np.int32) # Saved in int 16 but change to int32 to avoid overflow
                assert shift_id < 2 ** 31 # <- We have already run into problem in a past interation
                pred += shift_id
                max_id_seen = max(max_id_seen, pred.max())

            # As in semantic segmentation, 4D-PLS implementation filters out points marked 
            # with noisy / ignore semantic labels
            where_ignore = (labels_cls == 0)
            labels_inst = labels_inst[~where_ignore]
            pred = pred[~where_ignore]

            # Find points where there is no instance annotation and marked them with dedicated ID
            no_gt = ((labels_inst % 1000) == 0) # <- Use convention in nuscenes labels

            # Augment 'labels_inst' and 'pred' with idx of scene to avoid collisions of ids between scenes
            where_no_prediction = (pred == NO_PREDICTION_ID)
            pred = pred.astype(np.int64)
            labels_inst = labels_inst.astype(np.int64)

            # First 32 bits encode sequence index
            pred_temporal = (idx_seq << 32) + pred
            labels_inst_temporal = (idx_seq << 32) + labels_inst
            # For scanwise metric, use frame_idx instead of idx_seq                
            pred_scanwise = (frame_idx << 32) + pred
            labels_inst_scanwise = (frame_idx << 32) + labels_inst

            # Re-apply mask where there is no GT and no prediction
            labels_inst_temporal[no_gt] = NO_GROUND_TRUTH_ID
            pred_temporal[where_no_prediction] = NO_PREDICTION_ID
            labels_inst_scanwise[no_gt] = NO_GROUND_TRUTH_ID
            pred_scanwise[where_no_prediction] = NO_PREDICTION_ID

            # Compute scores for this frame
            lut_temporal_0 = s_assoc_quantities(labels_inst_temporal, pred_temporal, 0, lut_temporal_0, frame_idx)
            lut_temporal_50 = s_assoc_quantities(labels_inst_temporal, pred_temporal, 50, lut_temporal_50, frame_idx)
            lut_scanwise_0 = s_assoc_quantities(labels_inst_scanwise, pred_scanwise, 0, lut_scanwise_0, frame_idx)
            lut_scanwise_50 = s_assoc_quantities(labels_inst_scanwise, pred_scanwise, 50, lut_scanwise_50, frame_idx)

            # Next frame
            current_sample_token = current_sample["next"]

            # Increment frame idx
            frame_idx += 1

    # Final result
    print("Scores:")
    S_assoc, IoU_star = s_assoc(lut_temporal_0)
    print(
        f"Temporal S_assoc: {S_assoc:.5f}. Temporal IoU*: {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_temporal_50)
    print(
        f"Temporal S_assoc filtered (50 pts): {S_assoc:.5f}. Temporal IoU* filtered (50 pts): {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_scanwise_0)
    print(
        f"Scanwise S_assoc: {S_assoc:.5f}. Scanwise IoU*: {IoU_star:.5f} "
    )
    S_assoc, IoU_star = s_assoc(lut_scanwise_0)
    print(
        f"Scanwise S_assoc filtered (50 pts): {S_assoc:.5f}. Scanwise IoU* filtered (50 pts): {IoU_star:.5f} "
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./evaluate_panoptic.py")
    parser.add_argument(
        '--predictions',
        '-p',
        type=str,
        required=False,
        help='Prediction dir. Same organization as dataset, but predictions in'
        'each sequences "prediction" directory. No Default.')

    FLAGS, unparsed = parser.parse_known_args()
    # ---
    # NuScenes dataloader
    nusc = NuScenes(
        version='v1.0-trainval', 
        dataroot='/datasets_local/nuscenes/', 
        verbose=False,
    )
    compute_s_assoc_nuscenes(FLAGS.predictions, nusc)
