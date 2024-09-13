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
import numpy as np
from tqdm import tqdm
from functools import reduce
import pypatchworkpp as pwpp
from multiprocessing import Pool
from pyquaternion import Quaternion
from utils.utils import get_cpu_limit
from nuscenes.nuscenes import NuScenes
from preprocess.pcd_preprocess import clusterize_pcd, grid_sample, order_segments, apply_transform



DATA_DIR = "datasets/nuscenes/"
AUGMENTED_DIR = 'segments_gridsample'
DOWNSAMPLING_RESOLUTION = [0.05,0.05,0.05,5]
SCAN_WINDOW = 80


def create_list():
    nusc = NuScenes(
        version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
    )
    points_datapath = datapath_pretrain(nusc)
    del nusc
    os.makedirs(os.path.join(DATA_DIR, 'assets', AUGMENTED_DIR, "samples", "LIDAR_TOP"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'assets', AUGMENTED_DIR, "sweeps", "LIDAR_TOP"), exist_ok=True)
    return points_datapath


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                 rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                 inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def datapath_pretrain(nusc):
    points_datapath = []
    pdp = []
    for scene in nusc.scene:
        samples = []
        current_sample_token = scene["first_sample_token"]
        current_sample_token = nusc.get("sample", current_sample_token)["data"]["LIDAR_TOP"]
        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample_data = nusc.get("sample_data", current_sample_token)
            calibration = nusc.get('calibrated_sensor', current_sample_data['calibrated_sensor_token'])
            ref_pose = nusc.get('ego_pose', current_sample_data['ego_pose_token'])
            samples.append((current_sample_data, calibration, ref_pose))
            current_sample_token = current_sample_data["next"]
        pdp.append(samples)
    for seq in pdp:
        for i in range(0, len(seq), SCAN_WINDOW):
            points_datapath.append(seq[i:i + SCAN_WINDOW])
    return points_datapath


def aggregate_pcds_4d(data_batch):
    # load empty pcd point cloud to aggregate
    points_set = np.empty((0, 4))
    num_points = []
    distance = []

    calibration = data_batch[0][1]
    ref_from_car = transform_matrix(calibration['translation'], Quaternion(calibration['rotation']), inverse=True)
    car_from_current = transform_matrix(calibration['translation'], Quaternion(calibration['rotation']), inverse=False)
    ref_pose = data_batch[0][2]
    car_from_global = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)
    for t in range(len(data_batch)):
        fname = data_batch[t][0]['filename']
        # load the next t scan, apply pose and aggregate
        scan = np.fromfile(os.path.join(DATA_DIR, fname), dtype=np.float32)
        p_set = scan.reshape((-1, 5))[:, :4]
        p_set[:, 3] = t
        dist_ref = p_set[:, :2]
        # dist_ref[:, 0] *= 0.75
        distance.append(np.linalg.norm(dist_ref, axis=1))
        num_points.append(len(p_set))
        current_pose = data_batch[t][2]
        global_from_car = transform_matrix(current_pose['translation'],
                                                Quaternion(current_pose['rotation']), inverse=False)
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        p_set[:, :3] = apply_transform(p_set[:, :3], trans_matrix)
        points_set = np.vstack([points_set, p_set])
    return points_set, num_points, np.concatenate(distance)


def segment(frame):
    index, pdp = frame
    fname = pdp[-1][0]['filename']
    points_agg, num_points, distance = aggregate_pcds_4d(pdp)
    ground_label = np.empty(0, dtype=bool)
    data_batch = pdp
    params = pwpp.Parameters()
    params.sensor_height = 1.840
    params.th_seeds = 0.5
    params.th_dist = 0.25
    params.min_range = 2.
    params.num_rings_of_interest = 2
    params.num_sectors_each_zone = [8,16,27,16]
    PatchworkPLUSPLUS = pwpp.patchworkpp(params)
    for t in range(len(data_batch)):
        fname = data_batch[t][0]['filename']
        scan = np.fromfile(os.path.join(DATA_DIR, fname), dtype=np.float32)
        pointcloud = scan.reshape((-1, 5))[:, :4]
        PatchworkPLUSPLUS.estimateGround(pointcloud)
        g_set = np.full(pointcloud.shape[0], False, dtype=bool)
        g_set[PatchworkPLUSPLUS.getGroundIndices()] = True
        ground_label = np.concatenate([ground_label, g_set])

    points_ds, indexes, inverse_indexes = grid_sample(points_agg, DOWNSAMPLING_RESOLUTION, temporal_weight=0.03)
    ground_label = ground_label[indexes]
    del PatchworkPLUSPLUS
    points_ds[:, 2] = points_ds[:, 2] / 4. # For nuScenes to fix splicing
    segments = clusterize_pcd(points_ds, ground_label, min_samples=1, min_cluster_size=300)
    segments[np.logical_and(distance[indexes] < 2., ground_label != 1)] = -1
    segments_agg = segments[inverse_indexes]
    segments_agg = order_segments(segments_agg, min_points=50)
    idx = 0
    for frame, num in zip(pdp, num_points):
        fname = frame[0]['filename']
        fname = str(os.path.join(DATA_DIR, 'assets', AUGMENTED_DIR, fname))[:-4] + f"_{index}.seg"
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
