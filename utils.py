import torch
import random
import numpy as np
from collections import defaultdict
import json

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed_torch(3)
# print('Set seed to 3.')

def get_hierarchy_info(label_file):
    """
    :param label_file: the path of the label_cpt file
    :return: parent_to_children: Dict{str -> Set[str]}, the parent-child relationship of labels
    :return: label_to_id: Dict{str -> int}, the label to id mapping
    :return: child_to_parent: Dict{str -> str}, the child-parent relationship of labels
    :return: label_depth: Dict{str -> int}, the depth of each label
    """
    parent_to_children = defaultdict(set)
    label_to_id = {}
    with open(label_file) as f:
        label_to_id['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_to_id:
                    label_to_id[i] = len(label_to_id) - 1
                parent_to_children[line[0]].add(i)
        label_to_id.pop('Root')
    child_to_parent = {}
    for i in parent_to_children:
        for j in list(parent_to_children[i]):
            child_to_parent[j] = i

    def get_ancestors(label):
        if child_to_parent[label] != 'Root':
            return [label, ] + get_ancestors(child_to_parent[label])
        else:
            return [label]

    label_depth = {}
    for label in label_to_id:
        label_depth[label] = len(get_ancestors(label))
    
    return parent_to_children, label_to_id, child_to_parent, label_depth

def save_results(pred, gold, scores, label_dict, dev_input, epoch, path, threshold=0.5, top_k=None):
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']

    macro_precision = scores['macro_precision']
    micro_precision = scores['micro_precision']
    macro_recall = scores['macro_recall']
    micro_recall = scores['micro_recall']

    res = {'epoch': epoch,
           'macro_f1': macro_f1, 'micro_f1': micro_f1,
              'macro_precision': macro_precision, 'micro_precision': micro_precision,
                'macro_recall': macro_recall, 'micro_recall': micro_recall}
    
    assert(len(pred) == len(gold) == len(dev_input)), f'pred: {len(pred)}, gold: {len(gold)}, dev_input: {len(dev_input)}'

    predictions = []
    for pred, gold, input_text in zip(pred, gold, dev_input):
        np_sample_predict = np.array(pred, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict.flatten())
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(pred)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])
    
        predictions.append({'input': input_text, 'pred':[label_dict[int(i)] for i in sample_predict_id_list], 'gold':[label_dict[int(i)] for i in gold]})

    res['predictions'] = predictions
    
    with open(path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    