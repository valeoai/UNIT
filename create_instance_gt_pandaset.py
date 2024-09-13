# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


def points_in_boxes(point_cloud, boxes):
    """
    Vectorized computation to get a binary matrix indicating if a point is in a box.
    """
    centers = boxes[:, :3]
    lengths = boxes[:, 3:6] / 2
    yaws = boxes[:, 6]

    # Translate point cloud to the coordinate system of each box
    translated_point_cloud = point_cloud[:, np.newaxis, :] - centers

    # Rotate point cloud back to the axis-aligned orientation of each box
    rotation_matrices = np.array(list([[np.cos(-yaw), -np.sin(-yaw), 0],
                                  [np.sin(-yaw), np.cos(-yaw), 0],
                                  [0, 0, 1]] for yaw in yaws))
    rotated_point_cloud = np.einsum('ijk,jlk->ijl', translated_point_cloud, rotation_matrices)

    # Check if points lie within the box lengths
    in_box = np.abs(rotated_point_cloud) < lengths

    # Check if all dimensions are within the box
    in_box = np.all(in_box, axis=2)

    return in_box


class PandasetSemSeg(Dataset):

    MAPPING_CLASS = {
        0: 'Noise',
        1: 'Smoke',
        2: 'Exhaust',
        3: 'Spray or rain',
        4: 'Reflection',
        5: 'Vegetation',
        6: 'Ground',
        7: 'Road',
        8: 'Lane Line Marking',
        9: 'Stop Line Marking',
        10: 'Other Road Marking',
        11: 'Sidewalk',
        12: 'Driveway',
        13: 'Car',
        14: 'Pickup Truck',
        15: 'Medium-sized Truck',
        16: 'Semi-truck',
        17: 'Towed Object',
        18: 'Motorcycle',
        19: 'Other Vehicle - Construction Vehicle',
        20: 'Other Vehicle - Uncommon',
        21: 'Other Vehicle - Pedicab',
        22: 'Emergency Vehicle',
        23: 'Bus',
        24: 'Personal Mobility Device',
        25: 'Motorized Scooter',
        26: 'Bicycle',
        27: 'Train',
        28: 'Trolley',
        29: 'Tram / Subway',
        30: 'Pedestrian',
        31: 'Pedestrian with Object',
        32: 'Animals - Bird',
        33: 'Animals - Other',
        34: 'Pylons',
        35: 'Road Barriers',
        36: 'Signs',
        37: 'Cones',
        38: 'Construction Signs',
        39: 'Temporary Construction Barriers',
        40: 'Rolling Containers',
        41: 'Building',
        42: 'Other Static Object',
    }

    def __init__(self, rootdir, which_pandar, **kwargs):
        super().__init__()
        self.rootdir = rootdir

        # Select lidar
        assert which_pandar in ["pandar_64", "pandar_gt"]
        self.which_pandar = which_pandar
        self.select_pandar = 0 if which_pandar == "pandar_64" else 1

        # Class mapping
        self.mapping = np.vectorize(PandasetSemSeg.MAPPING_CLASS.__getitem__)

        # List of scenes
        scene_list = np.sort(glob(self.rootdir + "/*/annotations/semseg/"))
        scene_list = np.array([f.split("/")[-4] for f in scene_list])

        # Split
        assert len(scene_list) == 76
        scene_list = np.sort(scene_list)

        # Find all files
        self.im_idx, self.poses = [], []
        for scene_id in scene_list:
            self.im_idx.extend(
                np.sort(glob(self.rootdir + f"/{scene_id}/lidar/*.pkl.gz"))
            )
            file_pose = self.rootdir + f"/{scene_id}/lidar/poses.json"
            with open(file_pose, "r") as f:
                self.poses.extend(json.load(f))

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        # Load pc and labels
        pc = pd.read_pickle(self.im_idx[index]).values
        where_pandar = pc[:, -1] == self.select_pandar
        pc = pc[where_pandar, :-2]
        assert pc.shape[1] == 4

        # Load label
        label = pd.read_pickle(
            self.im_idx[index].replace("lidar", "annotations/semseg")
        ).values
        label = label[where_pandar, 0]
        
        # load boxes
        cuboids = pd.read_pickle(
            self.im_idx[index].replace("lidar", "annotations/cuboids")
        ).values
        # we want both -1 or self.select_pandar
        cuboids = cuboids[cuboids[:,13] != (1-self.select_pandar)]
        boxes = cuboids[:, [5,6,7,8,9,10,2]].astype(np.float64)
        boxes_class = cuboids[:, 1]
        boxes_id = cuboids[:, 0]
        boxes_id = np.vectorize(lambda x: abs(hash(x)))(boxes_id)
        points_id = np.zeros_like(label)
        points_class = self.mapping(label)
        binary_matrix = points_in_boxes(pc[:, :3], boxes)
        condition = points_class[:, None] == boxes_class
        binary_matrix = np.logical_and(binary_matrix, condition)
        # Find a match for each point
        matching_box = np.argmax(binary_matrix, 1)
        # If there is no match, set the box to -1
        matching_box[binary_matrix.sum(axis=1) == 0] = -1
        # Now find single matches
        single_matches = binary_matrix.sum(1) == 1
        # Box might be slightly off, so we need to check if we missed points
        # We do this by checking if the distance to the nearest point in the box is less than 1m
        for cl in np.unique(boxes_class):
            condition = boxes_class == cl
            # take the points assigned to exactly one box
            assigned_points = condition[matching_box] & single_matches & (matching_box != -1)
            # take the points that are not assigned to any box
            unassigned_points = np.logical_and(points_class == cl, ~assigned_points)
            # find the nearest assigned point for each unassigned point
            if assigned_points.sum():
                tree = cKDTree(pc[:, :3][assigned_points])
                dist, assignment = tree.query(pc[:, :3][unassigned_points])
                # if the nearest point is less than m away, assign it to the box
                assert points_id[assigned_points].sum() == 0
                assert points_id[unassigned_points].sum() == 0
                points_id[assigned_points] = boxes_id[matching_box[assigned_points]]
                idx = np.where(unassigned_points)[0][dist < 1.]
                points_id[idx] = boxes_id[matching_box[assigned_points][assignment]][dist < 1.]
        scene, _, file = self.im_idx[index].rsplit('/', 3)[1:]
        save_path = os.path.join(self.rootdir, "preprocessed", self.which_pandar, scene, "instances", file)
        os.makedirs(os.path.join(self.rootdir, "preprocessed", self.which_pandar, scene, "instances"), exist_ok=True)
        pd.DataFrame(points_id).to_pickle(save_path)
        return


class Pandaset64SemSeg(PandasetSemSeg):
    def __init__(self, **kwargs):
        super().__init__(which_pandar="pandar_64", **kwargs)


class PandasetGTSemSeg(PandasetSemSeg):
    def __init__(self, **kwargs):
        super().__init__(which_pandar="pandar_gt", **kwargs)


if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(description="Switch between Pandaset64SemSeg and PandasetGTSemSeg")
    parser.add_argument('--sensor', choices=['pandar64', 'pandargt'], default='pandargt',
                        help="Choose between the Pandar64 ('pandar64') and PandarGT ('pandargt') sensors. Default is 'pandargt'.")
    args = parser.parse_args()

    rootdir = "datasets/pandaset"
    os.makedirs(os.path.join(rootdir, "preprocessed"), exist_ok=True)
    # Instantiate the appropriate dataset class based on the argument
    if args.dataset == 'pandar64':
        print("Starting preprocessing for Pandar64")
        dataset = Pandaset64SemSeg(rootdir=rootdir)
    else:
        print("Starting preprocessing for PandarGT")
        dataset = PandasetGTSemSeg(rootdir=rootdir)

    # Process the dataset
    for pc in tqdm(dataset):
        pass
