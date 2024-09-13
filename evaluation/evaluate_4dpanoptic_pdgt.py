# This file is covered by the LICENSE file of https://github.com/PRBonn/semantic-kitti-ap
import argparse
import os
import glob
import pandas as pd
import numpy as np
import time
import json

from eval_np import Panoptic4DEval


# possible splits
splits = ["train", "val", "test"]



def get_pattern_and_precision(frame):
    segment, instance = frame
    if '_' in os.path.basename(segment):
        pattern = '_'
    else:
        pattern = '.'
    precision = np.int32
    seg = np.fromfile(segment, dtype=precision)
    labels = pd.read_pickle(instance).values
    if seg.shape[0] == labels.shape[0]:
        return pattern, precision
    else:
        precision = np.int16
        assert np.fromfile(segment, dtype=precision).shape[0] == labels.shape[0]
        return pattern, precision


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_panoptic.py")
  parser.add_argument(
      '--predictions',
      '-p',
      type=str,
      required=False,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default.')
  parser.add_argument(
      '--split',
      '-s',
      type=str,
      required=False,
      choices=splits,
      default="val",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit',
      '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--min_inst_points',
      type=int,
      required=False,
      default=50,
      help='Lower bound for the number of points to be considered instance',
  )
  parser.add_argument(
      '--precision',
      type=int,
      required=False,
      default=None,
      help='Output directory for scores.txt and detailed_results.html.',
  )

  start_time = time.time()

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Limit: ", FLAGS.limit)
  print("Min instance points: ", FLAGS.min_inst_points)
  print("*" * 80)

  # assert split
  assert (FLAGS.split in splits)

  # open data config file

  # get number of interest classes, and the label mappings
  # class
  nr_classes = 2

  class_lut = np.ones((2550), dtype=np.int32)
  class_lut[0] = 0

  # class
  ignore_class = [0]

  print("Ignoring classes: ", ignore_class)
  phase_scenes = np.sort(glob.glob("../datasets/pandaset/*/annotations/semseg/"))
  phase_scenes = np.array([f.split("/")[-4] for f in phase_scenes])
  pos_scene = []
  for scene in phase_scenes:
      file_pose = f"../datasets/pandaset/{scene}/meta/gps.json"
      with open(file_pose, "r") as f:
          gps = json.load(f)
      pos_scene.append(gps[0]["lat"])
  pos_scene = np.array(pos_scene)
    # Split
  val_split = pos_scene <= 37.6
  train_split = pos_scene > 37.6
  if FLAGS.split == "train":
      phase_scenes = phase_scenes[train_split]
      assert len(phase_scenes) == 49
  elif FLAGS.split == "val":
      phase_scenes = phase_scenes[val_split]
      assert len(phase_scenes) == 27
  else:
      raise Exception(f"Unknown split {FLAGS.split}")
  phase_scenes = np.sort(phase_scenes)

  # create evaluator
  class_evaluator_0 = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=0)
  class_evaluator_notemp_0 = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=0)
  class_evaluator = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)
  class_evaluator_notemp = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)

  # get label paths
  label_names = []
  pred_names = []
  pattern = None
  index = 0
  map_instances = {0:0} # in my GT PD seg, instance number are too big for the 4D hack of eval_np
  current_map = 1
  for sequence in phase_scenes:
    seq_label_names = ['../datasets/pandaset/preprocessed/pandar_gt/' + sequence + '/instances/' + str(id).zfill(2) + '.pkl.gz' for id in range(80)]
    seq_pred_names = sorted(glob.glob(FLAGS.predictions + '/' + sequence + '/lidar/' + '*.seg'))

    label_names.extend(seq_label_names)
    pred_names.extend(seq_pred_names)

  # check that I have the same number of files
  assert (len(label_names) == len(pred_names))
  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison

  complete = len(label_names)
  count = 0
  percent = 10
  context_window = None
  offset = 0
  max_value = 0
  for label_file, pred_file in zip(label_names, pred_names):
    count = count + 1
    if 100 * count / complete > percent:
      print("{}% ".format(percent), end="", flush=True)
      percent = percent + 10

    u_label_inst = pd.read_pickle(label_file).values[:, 0]
    for inst in np.unique(u_label_inst):
      if inst not in map_instances:
        map_instances[inst] = current_map
        current_map += 1
    u_label_inst = np.vectorize(map_instances.__getitem__)(u_label_inst).astype(np.int32)
    pc = pd.read_pickle(label_file.replace('preprocessed/pandar_gt/', '').replace('instances', 'lidar')).values
    where_pandar = pc[:, -1] == 1
    labels_class = pd.read_pickle(
        label_file.replace('preprocessed/pandar_gt/', '').replace('instances', 'annotations/semseg')
      ).values
    labels_class = labels_class[where_pandar, 0]

    u_label_sem_class = class_lut[labels_class.astype(np.int32)]

    if FLAGS.precision is None:
      if len(np.fromfile(pred_file, dtype=np.int16)) == len(labels_class):
        FLAGS.precision = np.int16
      else:
        FLAGS.precision = np.int32
    try:
      new_context_window = pred_file.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0]
      if new_context_window != context_window:
        context_window = new_context_window
        offset += max_value
        max_value = 0
      label = np.fromfile(pred_file, dtype=FLAGS.precision).astype(np.int32)
      max_value = max(max_value, label.max())
      label[label > 0] += offset
    except IndexError:
      label = np.fromfile(pred_file, dtype=FLAGS.precision).astype(np.int32)

    # label = np.fromfile(pred_file, dtype=np.uint32)

    # u_pred_sem_class = (label >= 0).astype(np.int32)
    u_pred_sem_class = np.ones_like(label)
    u_pred_inst = label + 2  # hack to consider all preds as instances
    if FLAGS.limit is not None:
      u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
      u_pred_inst = u_pred_inst[:FLAGS.limit]

    class_evaluator.addBatch(label_file.split('/')[-3], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_notemp.addBatch(label_file, u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_0.addBatch(label_file.split('/')[-3], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_notemp_0.addBatch(label_file, u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

  print("100%")

  complete_time = time.time() - start_time
  print("Scores with no filtering")
  LAQ_ovr, AQ_p, AQ_r, iou_star = class_evaluator_notemp_0.getPQ4D()
  print("S_assoc (LAQ):", LAQ_ovr)
  LAQ_ovr, AQ_p, AQ_r, iou_star = class_evaluator_0.getPQ4D()
  print("S_assoc_temporal (LAQ):", LAQ_ovr)
  print("IoU* (LAQ):", iou_star)

  print(f"Scores with {FLAGS.min_inst_points} points filtering")
  LAQ_ovr, AQ_p, AQ_r, iou_star = class_evaluator_notemp.getPQ4D()
  print("S_assoc (LAQ):", LAQ_ovr)
  LAQ_ovr, AQ_p, AQ_r, iou_star = class_evaluator.getPQ4D()
  print("S_assoc_temporal (LAQ):", LAQ_ovr)
  print("IoU* (LAQ):", iou_star)