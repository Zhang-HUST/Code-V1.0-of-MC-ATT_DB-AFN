import os
import math
import torch
import numpy as np
from trainTest.datasets.dataloader_shffule_utils import initDatasetShffule
from utils.common_utils import calculate_class_weights_torch


def get_fileName_weights(path, subject, subjects_list):
    if subject not in subjects_list:
        raise ValueError('subject not in subjects_list_global', subjects_list)
    encoded_label_name = 'sub_label_encoded'
    file_name = os.path.join(path, ''.join([subject, '_targetTrainData.npz']))
    with open(file_name, 'rb') as f:
        raw_labels = np.load(f)[encoded_label_name]
    raw_label_type = list(np.unique(raw_labels))
    class_weights = calculate_class_weights_torch(raw_labels)

    return file_name, class_weights, encoded_label_name, raw_label_type


def get_intra_dataloaders(path, label_name, total_exp_time, current_exp_time, train_batch, test_batch,
                          valid_batch, modal):
    init_dataset = initDatasetShffule()
    init_dataset.initIntraSubjectDataset(path=path, label_name=label_name, total_exp_time=total_exp_time, modal=modal)
    train_loader, valid_loader, test_loader = init_dataset.getDataLoader_intra(exp_time=current_exp_time,
                                                                               train_batch=train_batch,
                                                                               test_batch=test_batch,
                                                                               valid_batch=valid_batch)
    return train_loader, valid_loader, test_loader


def get_print_info(subjects_list):
    info = ['当前任务：下肢运动识别，总受试者：', subjects_list]
    return info


def get_save_path(base_path, model_name, subject):
    absolute_path = os.path.join(base_path, model_name, ''.join(['Sub', subject]))
    relative_path = os.path.relpath(absolute_path, base_path)
    return {'absolute_path': absolute_path, 'relative_path': relative_path}


def get_basic_node_amount(modal):
    if modal == 'E':
        basic_node_amount = 14
    elif modal == 'A':
        basic_node_amount = 15
    elif modal == 'E-A':
        basic_node_amount = 29
    else:
        raise ValueError('modal error!')
    return basic_node_amount

