# This file is covered by the LICENSE file of https://github.com/PRBonn/semantic-kitti-ap
import argparse
import os
import yaml
import numpy as np
import time

from eval_np import Panoptic4DEval


# possible splits
splits = ["train", "valid", "test"]

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
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg',
      '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti-unsup.yaml",
      help='Dataset config file. Defaults to %(default)s',
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
  )

  start_time = time.time()

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("Min instance points: ", FLAGS.min_inst_points)
  print("*" * 80)

  # assert split
  assert (FLAGS.split in splits)

  # open data config file
  DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))

  # get number of interest classes, and the label mappings
  # class
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)
  class_strings = DATA["labels"]

  # make lookup table for mapping
  # class
  maxkey = max(class_remap.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  class_lut = np.zeros((maxkey + 100), dtype=np.int32)
  class_lut[list(class_remap.keys())] = list(class_remap.values())

  # class
  ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

  print("Ignoring classes: ", ignore_class)

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # create evaluator
  class_evaluator_0 = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=0)
  class_evaluator_notemp_0 = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=0)
  class_evaluator = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)
  class_evaluator_notemp = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join("../datasets/semantickitti/dataset/sequences", sequence, "labels")
    # populate the label names
    seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
    label_names.extend(seq_label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, sequence)
    # populate the label names
    seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".seg")])
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
    # print("evaluating label ", label_file, "with", pred_file)
    # open label

    label = np.fromfile(label_file, dtype=np.uint32)

    u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format

    # ***** 
    # Orig. line
    # u_label_inst = label >> 16 # Instances of different classes can share the same ID
    # Corrected lines
    # u_label_inst = (label >> 16) + (u_label_sem_class << 16)
    u_label_inst = label # Make sure to have different IDs
    u_label_inst[(label >> 16) == 0] = 0 # Remap to zero where there is no instance annotations
    u_label_inst[u_label_sem_class == 0] = 0
    # Map all GT class to 1 (except where ignore) as we are in class-agnostic setting
    u_label_sem_class[u_label_sem_class > 0] = 1
    # *****
    if FLAGS.limit is not None:
      u_label_sem_class = u_label_sem_class[:FLAGS.limit]
      u_label_inst = u_label_inst[:FLAGS.limit]

    if FLAGS.precision is None:
      if len(np.fromfile(pred_file, dtype=np.int16)) == len(label):
        FLAGS.precision = np.int16
      else:
        FLAGS.precision = np.int32
    try:
      # This part is used for segments with context windows
      new_context_window = pred_file.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0]
      if new_context_window != context_window:
        context_window = new_context_window
        offset += max_value
        max_value = 0
      label = np.fromfile(pred_file, dtype=FLAGS.precision).astype(np.int32)
      max_value = max(max_value, label.max())
      label[label > 0] += offset
    except IndexError:
      label = np.fromfile(pred_file, dtype=FLAGS.precision).astype(np.uint32)

    # label = np.fromfile(pred_file, dtype=np.uint32)

    # ***** 
    # Orig. line
    # u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    # u_pred_inst = label >> 16
    # Corrected lines
    u_pred_sem_class = np.ones_like(label).astype(np.int32)  # Class-agnostic map sem class to 1
    u_pred_sem_class[label==0] = 0
    u_pred_inst = label
    # *****  
    if FLAGS.limit is not None:
      u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
      u_pred_inst = u_pred_inst[:FLAGS.limit]

    class_evaluator.addBatch(sequence, u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_notemp.addBatch(label_file.split('/')[-1], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_0.addBatch(sequence, u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
    class_evaluator_notemp_0.addBatch(label_file.split('/')[-1], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

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