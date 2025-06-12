# -*- coding: utf-8 -*-

"""
Utility functions for fake news detection
@Author             : Bao
@Date               : 2020/8/12
@Desc               : Common utilities for data processing and evaluation
@Last modified by   : Bao
@Last modified date : 2020/8/12
"""

import logging
import random
import numpy as np
import ujson as json
import torch
try:
    from .plm_checkers.checker_utils import soft_logic, soft_logic1
except:
    from plm_checkers.checker_utils import soft_logic, soft_logic1


def set_seed(args):
    """Set random seed for reproducibility"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_logger(level, filename=None, mode='a', encoding='utf-8'):
    """Initialize logger with specified configuration"""
    logging_config = {
        'format': '%(asctime)s - %(levelname)s - %(name)s:\t%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'level': level,
        'handlers': [logging.StreamHandler()]
    }
    if filename:
        logging_config['handlers'].append(logging.FileHandler(filename, mode, encoding))
    logging.basicConfig(**logging_config)


def read_json(filename, mode='r', encoding='utf-8'):
    """Read JSON file"""
    with open(filename, mode, encoding=encoding) as fin:
        return json.load(fin)


def save_json(data, filename, mode='w', encoding='utf-8'):
    """Save data to JSON file"""
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    """Read JSON lines file"""
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def save_json_lines(data, filename, mode='w', encoding='utf-8', skip=0):
    """Save data to JSON lines file"""
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(json.dumps(line, ensure_ascii=False), file=fout)


def read_json_dict(filename, mode='r', encoding='utf-8'):
    """Read JSON dictionary file and create reverse mapping"""
    with open(filename, mode, encoding=encoding) as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_json_dict(data, filename, mode='w', encoding='utf-8'):
    """Save dictionary to JSON file"""
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def get_prf(res):
    """Calculate precision, recall and F1 score"""
    if res['TP'] == 0:
        if res['FP'] == 0 and res['FN'] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res['TP'] / (res['TP'] + res['FP'])
        recall = 1.0 * res['TP'] / (res['TP'] + res['FN'])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_metrics(truth, predicted, z_predicted):
    """Compute evaluation metrics for fake news detection"""
    assert len(truth) == len(predicted)

    outputs = []
    results = {}
    cnt = 0  # Overall accuracy
    z_cnt_h, z_cnt_s = 0, 0   # Z-based accuracy (hard/soft)
    agree_h, agree_s = 0, 0   # Agreement between predictions
    y_zh = []
    y_zs = []
    y_zs_logit = []
    z_list = []
    
    for x, y, z in zip(truth, predicted, z_predicted):
        res = {'label': x, 'prediction': y}
        if x == y:
            cnt += 1

        res['pred_z'] = z

        # Soft logic computation
        y_ = soft_logic(torch.tensor([z]))[0]
        y_zs_logit.append(y_)

        ys_temp = y_.argmax(-1).item()
        y_zs.append(ys_temp)

        if ys_temp == x:
            z_cnt_s += 1
        if ys_temp == y:
            agree_s += 1

        # Hard logic computation
        z_h = torch.tensor(z[:]).argmax(-1).tolist()
        z_list.append(z_h)
        y__ = z_h
        y_zh.append(y__)

        if y__ == x:
            z_cnt_h += 1
        if y__ == y:
            agree_h += 1

        outputs.append(res)

    results['Accuracy'] = cnt / len(truth)
    results['z_Acc_hard'] = z_cnt_h / len(truth)
    results['z_Acc_soft'] = z_cnt_s / len(truth)
    results['Agreement_hard'] = agree_h / len(truth)
    results['Agreement_soft'] = agree_s / len(truth)
    
    return outputs, results, y_zh, z_list, y_zs, y_zs_logit
