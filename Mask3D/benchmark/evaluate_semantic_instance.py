import numpy as np


def s_assoc(inst_pred, inst_gt, labels_gt):
    s_assoc = 0.
    per_class_sassoc = dict()
    for i in np.unique(inst_gt)[1:]:  # remove unknown
        s = 0.
        ind_gt = np.where(inst_gt == i)[0]
        label_gt = labels_gt[ind_gt[0]]
        if label_gt not in per_class_sassoc:
            per_class_sassoc[label_gt] = []
        for j in np.unique(inst_pred):# [2:]:  # remove unknown and road
            ind_pred = np.where(inst_pred == j)[0]
            tpa = len(np.intersect1d(ind_gt, ind_pred))
            fpa = len(np.setdiff1d(ind_pred, ind_gt))
            fna = len(np.setdiff1d(ind_gt, ind_pred))
            iou = tpa / (tpa + fpa + fna)
            s += tpa * iou
        s_assoc += s / len(ind_gt)
        per_class_sassoc[label_gt].append(s / len(ind_gt))
    try:
        s_assoc /= len(np.unique(inst_gt)[1:])
    except ZeroDivisionError:
        s_assoc = 1.
    per_class_sassoc = {k: np.mean(v) for k, v in per_class_sassoc.items()}
    return s_assoc, per_class_sassoc


def s_assoc_conf(inst_pred, inst_gt):
    # version oftentimes faster
    # conf, _, _ = np.histogram2d(inst_pred, inst_gt, bins=[np.arange(max(inst_pred) + 2), np.arange(max(inst_gt) + 2)])
    # conf = conf.astype(int)

    # version more easy to understand, but usually slower
    conf = np.zeros((max(inst_pred)+1, max(inst_gt)+1), dtype=np.int32)
    np.add.at(conf, (inst_pred, inst_gt), 1)

    rows = conf.sum(axis=1)
    cols = conf.sum(axis=0)
    S = 0
    for j in range(1, conf.shape[1]):
        s_assoc = 0.
        for i in range(conf.shape[0]):
            if conf[i, j] > 0:
                s_assoc += conf[i, j] * (conf[i, j] / (rows[i] + cols[j] - conf[i, j]))
        if cols[j] > 0:
            S += s_assoc / cols[j]
    S = S / (len(np.unique(inst_gt)) - 1)
    s_assoc = np.mean(S)
    return s_assoc