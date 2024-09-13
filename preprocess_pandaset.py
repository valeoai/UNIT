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
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pypatchworkpp as pwpp
from multiprocessing import Pool
from pyquaternion import Quaternion
from utils.utils import get_cpu_limit
from preprocess.pcd_preprocess import clusterize_pcd, grid_sample, order_segments


DATA_DIR = "datasets/pandaset/"
AUGMENTED_DIR = 'segments_gridsample'
DOWNSAMPLING_RESOLUTION = [0.05,0.05,0.05,5]
SCAN_WINDOW = 40
WHICH_PANDAR = 1  # Put 1 for PandarGT and 0 for Pandar64


def heading_position_to_mat(heading, position):
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    transform_matrix = np.eye(4)
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    rotation = Quaternion(quat)
    transform_matrix[:3, :3] = rotation.rotation_matrix
    transform_matrix[:3, 3] = np.transpose(np.array(pos))
    return transform_matrix


def create_list():
    # List of scenes
    scene_list = np.sort(glob(DATA_DIR + "/*/lidar"))
    which_pandar = "pandar_64" if WHICH_PANDAR == 0 else "pandar_gt"
    points_datapath = []
    for scene in scene_list:
        scene_id = scene.split('/')[-2]
        os.makedirs(os.path.join(DATA_DIR, 'assets', which_pandar, AUGMENTED_DIR, scene_id, 'lidar'), exist_ok=True)
        for w in range(0, 80, SCAN_WINDOW):
            points_datapath.append([scene + f"/{i:02d}.pkl.gz" for i in range(w, w + SCAN_WINDOW)])
    return points_datapath


def aggregate_pcds_4d(frames):
    # load empty pcd point cloud to aggregate
    points_set = np.empty((0, 4))
    num_points = []

    for t, frame in enumerate(frames):
        p_set = pd.read_pickle(frame).values
        where_pandar = p_set[:, -1] == WHICH_PANDAR
        p_set = p_set[where_pandar, :4]
        # load the next t scan, apply pose and aggregate
        p_set[:, 3] = t
        num_points.append(len(p_set))
        points_set = np.vstack([points_set, p_set])
    return points_set, num_points


def segment(input):
    which_pandar = "pandar64" if WHICH_PANDAR == 0 else "pandargt"
    index, frames = input
    scene_id = frames[0].split('/')[-3]
    points_agg, num_points = aggregate_pcds_4d(frames)
    ground_label = np.empty(0, dtype=bool)
    params = pwpp.Parameters()
    params.sensor_height = 0.
    PatchworkPLUSPLUS = pwpp.patchworkpp(params)
    file_pose = os.path.join(DATA_DIR, scene_id, "lidar/poses.json")
    with open(file_pose, "r") as f:
        poses = json.load(f)
    for frame in frames:
        frame_id = frame.split('/')[-1].split('.')[0]
        pc = pd.read_pickle(frame).values
        where_pandar = pc[:, -1] == WHICH_PANDAR
        pc = pc[where_pandar, :4]
        # Transform to ego coordinate system
        pose = poses[int(frame_id)]
        transform_matrix = heading_position_to_mat(pose["heading"], pose["position"])
        transform_matrix = np.linalg.inv(transform_matrix)
        pc[:, :3] = pc[:, :3] @ transform_matrix[:3, :3].T
        pc[:, :3] += transform_matrix[:3, [3]].T

        # Extra shift along z-axis to have road approximately 1.7 m below the center of coord system
        pc[:, 2] -= 1.6
        PatchworkPLUSPLUS.estimateGround(pc)
        g_set = np.full(pc.shape[0], False, dtype=bool)
        g_set[PatchworkPLUSPLUS.getGroundIndices()] = True
        ground_label = np.concatenate([ground_label, g_set])
    points_ds, indexes, inverse_indexes = grid_sample(points_agg, DOWNSAMPLING_RESOLUTION, temporal_weight=0.03)
    ground_label = ground_label[indexes]
    del PatchworkPLUSPLUS
    segments = clusterize_pcd(points_ds, ground_label, min_samples=1, min_cluster_size=300)
    segments_agg = segments[inverse_indexes]
    segments_agg = order_segments(segments_agg, min_points=50)
    idx = 0
    for frame, num in zip(frames, num_points):
        fname = frame.replace('pandaset', f'pandaset/assets/{which_pandar}/{AUGMENTED_DIR}')
        fname = fname[:-7] + f"_{index}.seg"
        segments = segments_agg[idx:idx + num]
        segments.tofile(fname)
        idx += num


if __name__ == "__main__":
    points_datapath = create_list()
    frames = []
    for i, pdp in enumerate(points_datapath):
        frames.append((i, pdp))
    num_cpus = get_cpu_limit()
    print(f"Using {num_cpus} threads")
    with Pool(num_cpus) as p:
        list(tqdm(p.imap_unordered(segment, frames), total=len(frames)))
