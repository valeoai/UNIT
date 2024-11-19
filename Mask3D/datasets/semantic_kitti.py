import os
import logging
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
        dataset_name="semantic_kitti",
        data_dir: Optional[
            Union[str, Tuple[str]]
        ] = "data/processed/semantic_kitti",
        mode: Optional[str] = "train",
        add_reflection: Optional[bool] = True,
        add_distance: Optional[bool] = False,
        add_instance: Optional[bool] = True,
        add_raw_coordinates = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        sweep: Optional[int] = 1,
        filter_out_classes=[],
        label_offset=0,
        segments_dir=None,
        num_frames=1,
    ):
        self.dataset_name = dataset_name
        self.mode = mode
        data_dir = Path(data_dir) / "sequences"
        if mode == "train":
            self.data_dir = [data_dir / i / "velodyne" for i in ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]]
        elif mode == "validation":
            self.data_dir = [data_dir / i / "velodyne" for i in ["08",]]
        elif mode == "trainval":
            self.data_dir = [data_dir / i / "velodyne" for i in ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10", "08"]]
        elif mode == "debug":
            self.data_dir = [data_dir / i / "velodyne" for i in ["00",]]
        elif mode == "test":
            self.data_dir = [data_dir / i / "velodyne" for i in ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.ignore_label = ignore_label
        self.add_instance = add_instance
        self.add_distance = add_distance
        self.add_reflection = add_reflection
        self.add_raw_coordinates = add_raw_coordinates
        self.num_frames = num_frames

        self._labels = [{"name": "ignore"}, {"name": "road"}, {"name": "objects"}]

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        # loading database files
        self._data = []

        if mode == "test":
            assert num_frames == 1
            for database_path in self.data_dir:
                files = os.listdir(database_path)
                files = sorted(files)
                files = [database_path / file for file in files]
                for i in range(len(files)):
                    self._data.append([(files[i], None)])
        else:
            for database_path in self.data_dir:
                files = os.listdir(database_path)
                files = sorted(files)
                files = [database_path / file for file in files]
                segments_path = str(database_path).replace("sequences", f"assets/{segments_dir}")
                segments_path = segments_path.replace("velodyne", "")
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
        if mode == "debug":
            self._data = self._data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return_list = []
        for fname, lname in self.data[idx]:
            points = np.fromfile(fname, dtype=np.float32).reshape(-1,4)

            coordinates, features,  = (
                points[:, :3],
                points[:, 3:],
            )

            if lname is not None:
                instance = np.fromfile(lname, dtype=np.int16).astype(np.int32)
            else:
                instance = np.zeros((len(coordinates),), dtype=np.int32)

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

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    def _remap_from_zero(self, labels):
        labels += 1
        return labels
