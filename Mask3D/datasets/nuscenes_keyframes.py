# nuScenes dataloader on keyframes only (samples)

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple
import volumentations as V
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def fuzzy_search(list, pattern):
    for i in list:
        if pattern in i:
            return int(i.rsplit('_', 1)[1].split('.')[0])
    raise FileNotFoundError


class LidarDataset(Dataset):
    def __init__(
        self,
        dataset_name="nuscenes_keyframes",
        data_dir: Optional[
            Union[str, Tuple[str]]
        ] = "data/processed/nuscenes",
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
        self.data_dir = data_dir
        self.mode = mode
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=data_dir, verbose=False
        )
        if mode == "validation":
            mode = "val"
        phase_scenes = create_splits_scenes()[mode]
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

        listdir = os.listdir(Path(data_dir) / "assets" / segments_dir / 'samples' / 'LIDAR_TOP')
        index = 0
        for scene in nusc.scene:
            if scene["name"] not in phase_scenes:
                continue
            files = []
            labels = []
            current_sample_token = scene["first_sample_token"]  

            # Loop to get all successive keyframes
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                current_sample_data = nusc.get("sample_data", current_sample["data"]["LIDAR_TOP"])
                files.append(Path(data_dir) / current_sample_data['filename'])

                if f"{current_sample_data['filename'][18:-4]}_{index}.seg" in listdir:
                    labels.append(Path(data_dir) / "assets" / segments_dir / f"{current_sample_data['filename'][:-4]}_{index}.seg")
                elif f"{current_sample_data['filename'][18:-4]}_{index + 1}.seg" in listdir:
                    labels.append(Path(data_dir) / "assets" / segments_dir / f"{current_sample_data['filename'][:-4]}_{index + 1}.seg")
                    index = index + 1
                else:
                    index = fuzzy_search(listdir, current_sample_data['filename'][18:-4])
                    labels.append(Path(data_dir) / "assets" / segments_dir / f"{current_sample_data['filename'][:-4]}_{index}.seg")

                current_sample_token = current_sample["next"]
            for i in range(len(files) - num_frames + 1):
                id = int(str(labels[i]).split("_")[-1].split(".")[0])
                if int(str(labels[i + num_frames - 1]).split("_")[-1].split(".")[0]) == id:
                    ls = []
                    for j in range(num_frames):
                        ls.append((files[i+j], labels[i+j]))
                    self._data.append(ls)
        del nusc

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
            points = np.fromfile(fname, dtype=np.float32).reshape(-1,5)[:, :4]

            coordinates, features, = (
                points[:, :3],
                points[:, 3:],
            )

            instance = np.fromfile(lname, dtype=np.int16).astype(np.int32)

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
            # fname = "/".join(split[-3:])
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

    def _remap_from_zero(self, labels):
        labels += 1
        return labels