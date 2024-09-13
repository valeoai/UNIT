import os
import glob
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import volumentations as V
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LidarDataset(Dataset):
    def __init__(
        self,
        dataset_name="pandaset",
        data_dir: Optional[
            Union[str, Tuple[str]]
        ] = "data/processed/pandaset",
        mode: Optional[str] = "train",
        add_reflection: Optional[bool] = True,
        add_distance: Optional[bool] = False,
        add_instance: Optional[bool] = True,
        add_raw_coordinates = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        filter_out_classes=[],
        label_offset=0,
        segments_dir=None,
        num_frames=1,
    ):
        self.dataset_name = dataset_name        
        self.which_pandar = 1
        self.data_dir = data_dir
        phase_scenes = np.sort(glob.glob(data_dir + "/*/annotations/semseg/"))
        phase_scenes = np.array([f.split("/")[-4] for f in phase_scenes])
        # GPS coordinates
        pos_scene = []
        for scene in phase_scenes:
            file_pose = data_dir + f"/{scene}/meta/gps.json"
            with open(file_pose, "r") as f:
                gps = json.load(f)
            pos_scene.append(gps[0]["lat"])
        pos_scene = np.array(pos_scene)

        val_split = pos_scene <= 37.6
        train_split = pos_scene > 37.6
        self.mode = mode
        if mode == "train":
            phase_scenes = phase_scenes[train_split]
        elif mode == "validation":
            phase_scenes = phase_scenes[val_split]
        elif mode == "predict":
            phase_scenes = phase_scenes[val_split]
        elif mode == "test":
            raise Exception("Test mode not implemented for Pandaset")

        self.ignore_label = ignore_label
        self.add_instance = add_instance
        self.add_distance = add_distance
        self.add_reflection = add_reflection
        self.add_raw_coordinates = add_raw_coordinates
        self.num_frames = num_frames

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        # loading database files
        self._data = []

        for scene in phase_scenes:
            files = [f"{data_dir}/{scene}/lidar/{i:02d}.pkl.gz" for i in range(80)]
            segments_path = f"{data_dir}/assets/pandar_gt/{segments_dir}/{scene}/lidar"
            labels = os.listdir(segments_path)
            labels = sorted(labels)
            labels = [Path(segments_path) / label for label in labels]
            # This code works for the multiframe instead
            for i in range(len(files) - num_frames + 1):
                id = int(str(labels[i]).split("_")[-1].split(".")[0])
                if int(str(labels[i + num_frames - 1]).split("_")[-1].split(".")[0]) == id:
                    ls = []
                    for j in range(num_frames):
                        ls.append((files[i+j], labels[i+j]))
                    self._data.append(ls)

        # augmentations
        if num_frames == 1:
            self.volume_augmentations = V.NoOp()
            if volume_augmentations_path is not None:
                self.volume_augmentations = V.load(
                    volume_augmentations_path, data_format="yaml"
                )

        if data_percent < 1.0:
            # self._data = self._data[: int(len(self._data) * data_percent)]
            self._data = self._data[::round(1/data_percent)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return_list = []
        for fname, lname in self.data[idx]:
            points = pd.read_pickle(fname).values
            where_pandar = points[:, -1] == self.which_pandar  # Pandar64
            points = points[where_pandar, :4]

            coordinates, features, = (
                points[:, :3],
                points[:, 3:],
            )

            instance = np.fromfile(lname, dtype=np.int16)

            if not self.add_reflection:
                features = np.ones(np.ones((len(coordinates), 1)))

            if self.add_distance:
                center_coordinate = coordinates.mean(0)
                features = np.hstack(
                    (
                        features,
                        np.linalg.norm(coordinates - center_coordinate, axis=1)[
                            :, np.newaxis
                        ],
                    )
                )
            if self.add_raw_coordinates:
                features = np.hstack((features, coordinates))
            
            raw_coordinates = coordinates.copy()  # TODO check if this is necessary

            # prepare labels and map from 0 to 20(40)
            labels = instance.copy()
            labels += 1
            labels[labels>1] = 2

            labels = np.concatenate((labels[:, None], instance[:, None]), axis=1)

            # volume and image augmentations for train
            if "train" in self.mode and self.num_frames == 1:
                coordinates -= coordinates.mean(0)
                aug = self.volume_augmentations(
                    points=coordinates,
                    features=features,
                    labels=labels,
                )
                coordinates, features, labels = (
                    aug["points"],
                    aug["features"],
                    aug["labels"],
                )

            split = str(fname).split("/")
            fname = split[-3] + "/" + split[-1]
            return_list.append([
                coordinates, 
                features, 
                labels, 
                fname,
                raw_coordinates,
                idx
            ])
        return return_list

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def _remap_from_zero(self, labels):
        labels += 1

        return labels