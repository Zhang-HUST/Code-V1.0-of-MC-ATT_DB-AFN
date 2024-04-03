import math
import numpy as np
from scipy.signal import butter, lfilter, iirfilter
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from utils.common_utils import all_elements_equal_to_str, is_nan_in_df_rows, analyze_list, get_middle_value_in_list

### 通用工具
"""emg滤波器：陷波滤波、带通滤波、低通滤波"""


class emg_filtering():
    def __init__(self, fs, lowcut, highcut, imf_band, imf_freq):
        self.fs = fs
        # butterWorth带通滤波器
        self.lowcut, self.highcut = lowcut, highcut
        # 50 Hz陷波滤波器
        self.imf_band, self.imf_freq = imf_band, imf_freq
        # 低通滤波
        self.cutoff = 20

    def Implement_Notch_Filter(self, data, order=2, filter_type='butter'):
        # Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered
        fs = self.fs
        nyq = fs / 2.0
        freq, band = self.imf_freq, self.imf_band
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)

        return filtered_data

    def butter_bandpass(self, order=6):
        lowcut, highcut, fs = self.lowcut, self.highcut, self.fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        y = lfilter(b, a, data)

        return y

    def butter_lowpass(self, order=5):
        cutoff, fs = self.cutoff, self.fs
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return b, a

    def butter_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        y = lfilter(b, a, data)

        return y


"""多模态多通道数据归一化方法，其中支持归一化方法：'min-max'、'max-abs'、'positive_negative_one；归一化层面：'matrix'、'rows'"""


def data_nomalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            # 实例化 MinMaxScaler 并设置归一化范围为 [0, 1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            # 实例化 MinMaxScaler 并设置归一化范围为 [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            # print(np.min(scaler.data_min_),np.max(scaler.data_max_))
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            # 实例化 MaxAbsScaler，并拟合数据以计算每列的最大值和最小值的绝对值
            scaler = MaxAbsScaler()
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 将数据整体缩放到 [-1, 1] 范围内
            normalized_data = data / np.max(np.abs(data))
        else:
            print('Error: 未识别的normalize_method！')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            print('Error: 未识别的normalize_method！')
    else:
        print('Error: 未识别的normalize_level！')

    return normalized_data


### 对运动模式分类任务

"""基于滑动重叠窗口采样的样本集分割：重叠窗长window、步进长度step"""


def movement_classification_sample_segmentation(emg_data, imu_data, angle_data, label, window, step, classes,
                                                discard_transition_sample=True, verbose=True):
    emg_sample, imu_sample, angle_sample, label_encoded = [], [], [], []

    for i in range(classes):
        index = []
        for j in range(label.shape[0]):
            if label[j] == i:
                index.append(j)
        if len(index) != 0:
            if not discard_transition_sample:
                pass
            else:
                # index = index[window + step*(discard_transition_sample-1):-(window + step*(discard_transition_sample-1))]
                index = index[window:-window]
            emg = emg_data[index, :]
            imu = imu_data[index, :]
            angle = angle_data[index, :]
            length = math.floor((emg.shape[0] - window) / step)
            if verbose:
                print("class ", i, " number of sample: ", length)
            for j in range(length):
                sub_emg = emg[step * j:(window + step * j), :]
                emg_sample.append(sub_emg)
                sub_imu = imu[step * j:(window + step * j), :]
                imu_sample.append(sub_imu)
                sub_angle = angle[step * j:(window + step * j), :]
                angle_sample.append(sub_angle)
                label_encoded.append(i)
    emg_sample, imu_sample = np.array(emg_sample), np.array(imu_sample)
    angle_sample, label_encoded = np.array(angle_sample), np.array(label_encoded)
    if verbose:
        print('emg_sample: ', emg_sample.shape, ', imu_sample: ', imu_sample.shape)
        print('angle_sample: ', angle_sample.shape, ', label_encoded: ', label_encoded.shape)

    return emg_sample, imu_sample, angle_sample, label_encoded


def tj_movement_classification_sample_segmentation(emg_data, imu_data, angle_data, label, window, step,
                                                 verbose=True):
    emg_sample, imu_sample, angle_sample, label_encoded = [], [], [], []
    emg = emg_data
    imu = imu_data
    angle = angle_data
    length = math.floor((emg.shape[0] - window) / step)
    if verbose:
        print("class ", label, " number of sample: ", length)
    for j in range(length):
        sub_emg = emg[step * j:(window + step * j), :]
        emg_sample.append(sub_emg)
        sub_imu = imu[step * j:(window + step * j), :]
        imu_sample.append(sub_imu)
        sub_angle = angle[step * j:(window + step * j), :]
        angle_sample.append(sub_angle)
        label_encoded.append(label)
    emg_sample, imu_sample = np.array(emg_sample), np.array(imu_sample)
    angle_sample, label_encoded = np.array(angle_sample), np.array(label_encoded)
    if verbose:
        print('emg_sample: ', emg_sample.shape, ', imu_sample: ', imu_sample.shape)
        print('angle_sample: ', angle_sample.shape, ', label_encoded: ', label_encoded.shape)

    return emg_sample, imu_sample, angle_sample, label_encoded