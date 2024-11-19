
import re
import os
from os.path import basename, splitext
import numpy as np
import pypatchworkpp as pwpp
from torch.utils.data import Dataset
from preprocess.pcd_preprocess import clusterize_pcd, grid_sample, order_segments, apply_transform


LEARNING_MAP = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

class KITTIPreprocessor(Dataset):
    def __init__(self, data_dir, scan_window, split, downsampling_resolution, ground_method):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_gridsample'

        self.downsampling_resolution = downsampling_resolution
        self.scan_window = scan_window
        self.sampling_window = scan_window

        self.split = split
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif split == 'val':
            self.seqs = ['08']
        elif split == 'trainval':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.ground_method = ground_method

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_pretrain()
        self.nr_data = len(self.points_datapath)
        print('The size of %s data is %d' % (self.split, self.nr_data))

    def datapath_pretrain(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()

            for file_num in range(0, len(point_seq_bin), self.sampling_window):
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file]])
                if end_file == len(point_seq_bin):
                    break

    def __getitem__(self, index):
        seq_num = self.points_datapath[index][0].split('/')[-3]
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        # cluster the aggregated pcd and save the result
        cluster_path = os.path.join(self.data_dir, 'assets', self.augmented_dir, seq_num)
        if os.path.isfile(os.path.join(cluster_path, fname + f'_{index}.seg')):
            pass
        else:
            points_agg, num_points, distance = self.aggregate_pcds(self.points_datapath[index])
            if self.ground_method == "patchwork":
                ground_label = np.empty(0, dtype=bool)
                data_batch = self.points_datapath[index]
                for t in range(len(data_batch)):
                    fname = data_batch[t].split('/')[-1].split('.')[0]
                    # load the next t scan and aggregate
                    g_set = np.fromfile(os.path.join(self.data_dir, 'assets', 'patchwork', seq_num, fname + '.label'), dtype=np.uint32)
                    g_set = g_set == 1
                    ground_label = np.concatenate([ground_label, g_set])
                points_ds, indexes, inverse_indexes = grid_sample(points_agg, self.downsampling_resolution, temporal_weight=0.03)
                ground_label = ground_label[indexes]
            elif self.ground_method == "patchworkpp":
                # Patchwork++ not tried for the publication. Use at your own risk
                ground_label = np.empty(0, dtype=bool)
                data_batch = self.points_datapath[index]
                params = pwpp.Parameters()
                params.sensor_height = 0.
                PatchworkPLUSPLUS = pwpp.patchworkpp(params)
                datapath = data_batch[0].split('velodyne')[0]
                poses = self.load_poses(os.path.join(datapath, 'calib.txt'), os.path.join(datapath, 'poses.txt'))
                for t in range(len(data_batch)):
                    fname = data_batch[t].split('/')[-1].split('.')[0]
                    # load the next t scan, apply pose and aggregate
                    pc = np.fromfile(data_batch[t], dtype=np.float32)
                    pc = pc.reshape((-1, 4))
                    # Transform to ego coordinate system
                    pose_idx = int(fname)
                    pc[:, :3] = apply_transform(pc[:, :3], poses[pose_idx])
                    PatchworkPLUSPLUS.estimateGround(pc)
                    g_set = np.full(pc.shape[0], False, dtype=bool)
                    g_set[PatchworkPLUSPLUS.getGroundIndices()] = True
                    ground_label = np.concatenate([ground_label, g_set])
                points_ds, indexes, inverse_indexes = grid_sample(points_agg, self.downsampling_resolution, temporal_weight=0.03)
                ground_label = ground_label[indexes]
                del PatchworkPLUSPLUS

            labels = 1 - ground_label.copy().astype(np.int32)
            labels[np.logical_and(distance[indexes] < 2.2, ground_label != 1)] = -1

            segments = clusterize_pcd(points_ds, ground_label, min_samples=1, min_cluster_size=300)

            # filter based on distance to remove ego-vehicle pattern
            segments[np.logical_and(distance[indexes] < 2.2, ground_label != 1)] = -1

            segments_agg = segments[inverse_indexes]
            segments_agg = order_segments(segments_agg, min_points=50)

            idx = 0
            for fname, num in zip(self.points_datapath[index], num_points):
                segments = segments_agg[idx:idx + num]
                fname = splitext(basename(fname))[0]
                if not os.path.isdir(cluster_path):
                    os.makedirs(cluster_path)
                segments.tofile(os.path.join(cluster_path, fname + f'_{index}.seg'))
                idx += num

    def __len__(self):
        return self.nr_data

    def aggregate_pcds(self, data_batch):
        # load empty pcd point cloud to aggregate
        points_set = np.empty((0, 4))
        num_points = []
        distance = []

        fname = data_batch[0].split('/')[-1].split('.')[0]

        # load poses
        datapath = data_batch[0].split('velodyne')[0]
        poses = self.load_poses(os.path.join(datapath, 'calib.txt'), os.path.join(datapath, 'poses.txt'))

        for t in range(len(data_batch)):
            fname = data_batch[t].split('/')[-1].split('.')[0]
            # load the next t scan, apply pose and aggregate
            p_set = np.fromfile(data_batch[t], dtype=np.float32)
            p_set = p_set.reshape((-1, 4))
            p_set[:, 3] = t
            dist_ref = p_set[:, :2] - [0.5,0.]
            dist_ref[:, 0] *= 0.75
            distance.append(np.linalg.norm(dist_ref, axis=1))
            # distance.append(np.linalg.norm(p_set[:, :2] - [0.5,0.], axis=1))
            num_points.append(len(p_set))
            pose_idx = int(fname)
            p_set[:, :3] = apply_transform(p_set[:, :3], poses[pose_idx])
            points_set = np.vstack([points_set, p_set])

        return points_set, num_points, np.concatenate(distance)

    def load_poses(self, calib_fname, poses_fname):
        calibration = self.parse_calibration(calib_fname)
        poses_file = open(poses_fname)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []

        for line in poses_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses
    
    @staticmethod
    def parse_calibration(filename):
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib
