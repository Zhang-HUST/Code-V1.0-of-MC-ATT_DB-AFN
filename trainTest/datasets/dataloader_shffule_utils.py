import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

'''
    类initDataset的作用：
        initIntroSubjectDataset
        initInterSubjectDataset
    初始化时需传递的参数及其含义参数：
        initIntroSubjectDataset:
            path                 : 受试者npz文件路径
            raw_data_list        : 模态信号名称列表，以npz文件的字典规则为依据
            raw_feature_list     : 模态信号特征名称列表,可以为[]，表示不适用手动提取的特征
            label_name           : 用作label的npz文件字段名 ，应该是'..._encoded'
            total_exp_time       : 设计的总重复试验次数，步态相位识别中不起作用
            gait_or_motion       : 'motion':针对运动的，'gait'：针对步态的
            说明：
                initInroSubjectDataset等于给出了针对某一确定受试者提取train，valid，test DataSet的方法。当gait_or_motion参数为'motion'时，使用sklearn中的StratifiedShuffleSplit方法划分成
                total_exp_time组（train valid test）。  而当gait_or_motion参数为'gait'时，由于定义好了group_label 1-7 为训练， 8为验证，9-10为测试，因此只有一组通用的数据集（train valid test）。

        initInterSubjectDataset:


            说明：
        getDataLoader:
            exp_time             : 当前为第几次重复试验 ：1~total_exp_time
            train_batch          : 训练集batchSize
            test_batch           : 测试集batchSize
            valid_batch          : 验证集batchSize
'''


class initDatasetShffule:
    # 针对intra任务，读取单个受试者的npz文件，划分训练，测试，验证样本，供getDataLoader生成用于pytorch模型训练的dataLoader
    def initIntraSubjectDataset(self, path, label_name, total_exp_time, modal):
        # 函数返回 train valid test DataSet数据，同时也直接保存在self中
        if modal == 'E':
            data_list = ['sub_emg_sample']
        elif modal == 'A':
            data_list = ['sub_imu_sample']
        elif modal == 'G':
            data_list = ['sub_angle_sample']
        elif modal == 'E-A':
            data_list = ['sub_emg_sample', 'sub_imu_sample']
        elif modal == 'E-G':
            data_list = ['sub_emg_sample', 'sub_angle_sample']
        else:
            raise ValueError('modal error')

        test_ratio = 0.2
        valid_ratio = 0.1 / (1 - test_ratio)

        data = np.load(path)
        raw_data_container = []
        if len(data_list) == 0:
            raise Exception(f"you dont contain raw_time_domain_data,this is not right, check!")
        # 先将各模态信号在通道层拼接，可以通过raw_data_list选择信号模态
        for name in data_list:
            raw_data_container.append(data[name])
        raw_data = np.concatenate(raw_data_container, axis=1)
        self.raw_data_time_step = raw_data.shape[2]  # 记录原始时域信号的样本点（时间步_time_step）
        self.total_data = raw_data
        self.total_label = data[label_name]
        # 程序运行到这里，total_data,total_label就已经形成了
        self.total_data_shape = self.total_data.shape  # 记录一下shape
        # self.total_data = self.total_data.reshape(self.total_data_shape[0], -1)
        # self.total_label_shape = self.total_label.shape  # 记录一下shape
        # print(self.total_data.shape, self.total_label.shape)
        # 使用sklearn将数据集按照比例7：1：2划分成train：valid：test。
        self.X_train = []
        self.X_valid = []
        self.X_test = []
        self.y_train = []
        self.y_valid = []
        self.y_test = []
        sss = StratifiedShuffleSplit(n_splits=total_exp_time, test_size=test_ratio, random_state=42)
        sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=42)
        for train_index, test_index in sss.split(self.total_data, self.total_label):
            X_trainAndValid, X_test = self.total_data[train_index], self.total_data[test_index]
            y_trainAndValid, y_test = self.total_label[train_index], self.total_label[test_index]
            self.X_test.append(X_test.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.y_test.append(y_test)
            # 再将X_trainAndValid和Y_trainAndValid种划分出train和valid
            for train_index_, valid_index in sssForValid.split(X_trainAndValid, y_trainAndValid):
                X_train, X_valid = X_trainAndValid[train_index_], X_trainAndValid[valid_index]
                y_train, y_valid = y_trainAndValid[train_index_], y_trainAndValid[valid_index]
            self.X_train.append(X_train.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.X_valid.append(X_valid.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.y_train.append(y_train)
            self.y_valid.append(y_valid)
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    # 作用：返回三种DataSet：trainSet validSet testSet
    def getDataLoader_intra(self, exp_time, train_batch, valid_batch, test_batch):
        train_set = myDataset(data=self.X_train[exp_time - 1], label=self.y_train[exp_time - 1],
                              time_step=self.raw_data_time_step)
        valid_set = myDataset(data=self.X_valid[exp_time - 1], label=self.y_valid[exp_time - 1],
                              time_step=self.raw_data_time_step)
        test_set = myDataset(data=self.X_test[exp_time - 1], label=self.y_test[exp_time - 1],
                             time_step=self.raw_data_time_step)
        train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=valid_batch, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False)
        return train_loader, valid_loader, test_loader



class myDataset(Dataset):
    def __init__(self, data, label, time_step):
        self.data = data
        self.label = label
        self.time_step = time_step

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        singleData = torch.from_numpy(self.data[item, :, :]).unsqueeze(0)
        singleLabel = torch.from_numpy(self.label[item:item + 1]).to(dtype=torch.long).view(-1)
        return singleData, singleLabel


'''
    RepeatDataLoaderIterator:
        

'''


class RepeatDataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.dataset = data_loader.dataset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)

        return batch


if __name__ == '__main__':
    # 下面是一个应用例子
    # intro-subject-gait-recognition
    personOrder = '01'
    path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + personOrder + "_targetTrainData.npz"
    raw_data_list = ['sub_emg_sample', 'sub_angle_sample']
    raw_feature_list = ['sub_emg_features', 'sub_angle_features']
    label_name = 'sub_gait_label_encoded'
    gait_or_motion = 'gait'

    total_exp_time = 6
    train_batch = 32
    valid_batch = 32
    test_batch = 32
    exp_time = 6
    init_dataset = initDatasetShffule()
    init_dataset.initIntraSubjectDataset(path=path, raw_data_list=raw_data_list, raw_feature_list=[],
                                         label_name=label_name, total_exp_time=6, gait_or_motion=gait_or_motion)
    train_loader, valid_loader, test_loader = init_dataset.getDataLoader_intra(exp_time=exp_time,
                                                                               train_batch=train_batch,
                                                                               test_batch=test_batch,
                                                                               valid_batch=valid_batch)

    # inter-subject-gait-regocnition
    source_personOrder = ['01', '02', '03']
    target_personOrder = ['04']
    source_path_list = []
    target_path_list = []
    for order in source_personOrder:
        path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + order + "_targetTrainData.npz"
        source_path_list.append(path)
    for order in target_personOrder:
        path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + order + "_targetTrainData.npz"
        target_path_list.append(path)
    raw_data_list = ['sub_emg_sample', 'sub_angle_sample']
    raw_feature_list = ['sub_emg_features', 'sub_angle_features']
    label_name = 'sub_gait_label_encoded'
    gait_or_motion = 'gait'
    total_exp_time = 6
    exp_time = 6
    init_dataset = initDatasetShffule()
    init_dataset.initInterSubjectDataset(source_path_list=source_path_list, target_path_list=target_path_list,
                                         raw_data_list=raw_data_list, raw_feature_list=[], label_name=label_name,
                                         total_exp_time=6, gait_or_motion=gait_or_motion)
    source_train_loader, source_valid_loader, source_test_loader, target_train_loader, target_valid_loader, target_test_loader = init_dataset.getDataLoader_inter(
        exp_time=exp_time, source_train_batch=32, source_valid_batch=32, source_test_batch=32, target_train_batch=32,
        target_valid_batch=32, target_test_batch=32)

    # 调用RepeatDataLoaderIterator的例子：
    target_train_loader = RepeatDataLoaderIterator(target_train_loader)
    for data1, data2 in zip(source_train_loader, target_train_loader):
        print(data2[0].shape)
