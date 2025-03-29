import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pandas as pd
import os
import pywt
from scipy.signal import filtfilt

dataset_dir = './qt-database/'

patient_list = os.listdir(dataset_dir)
patient_list = [file[:-4] for file in patient_list if file.endswith(".dat")] 


# symbol을 통해서 QT 라벨링을 찾는 함수
# QT의 모양은 (N)t)와 (N)(t) 2가지라 2가지를 다 고려
# 추가적으로 data의 길이를 라벨이 시작하는 부분부터 끝나는 부분까지로 조정
def get_QT_labels(inds, symbols, data):
    QT_inds = []
    ind = 0
    N = len(symbols)
    while (ind < N - 4):
        if symbols[ind] == '(':
            if symbols[ind+1] == 'N':
                if symbols[ind + 3] == 't':
                    if symbols[ind+4] == ')':
                        tmp_inds = [i for i in range(inds[ind], inds[ind + 4] + 1)]
                        QT_inds.extend(tmp_inds)
                elif symbols[ind + 4] == 't':
                    if symbols[ind+5] == ')':
                        tmp_inds = [i for i in range(inds[ind], inds[ind + 5] + 1)]
                        QT_inds.extend(tmp_inds)

        ind = ind + 1

    data = data[QT_inds[0]:QT_inds[-1] + 1]
    QT_inds = np.array(QT_inds)
    QT_inds = QT_inds - QT_inds[0]

    return QT_inds, data


labeled_data = []

for record_name in patient_list:
    # 라벨링이 명확하게 안되어있어 제거한 레코드
    delete_record = ['sel102','sel116', 'sel213', 'sel221', 'sel232', 'sel233', 'sel308', 'sel35', 'sel37', 'sel44', 'sel46', 'sel808', 'sel891','sel820', 'sel847', 'sele0129']
    # 라벨링이 비어있는 부분이 많아 제거한 레코드
    test_record = ['sel14046','sel15814', 'sel223', 'sel301', 'sel302', 'sel310', 'sel803', 'sel871', 'sel873',
                   'sel104', 'sel114', 'sel14157', 'sel230', 'sel306', 'sel49', 'sel840', 'sele0107', 'sele0124', 'sele0166', 'sele0406',
                   'sel14172', 'sel231', 'sel50', 'sele0112', 'sel103', 'sel117']
    
    if record_name in delete_record:
        continue
    
    if record_name in test_record:
        continue
    
    
    # 주석 읽기
    annotation = wfdb.rdann(dataset_dir+record_name, 'q1c')

    record = wfdb.rdrecord(dataset_dir+record_name)
    annotation = wfdb.rdann(dataset_dir+record_name, 'q1c')
    
    # signal 2의 경우 라벨이 제대로 되어있는 부분이 많이 없어서 배제하고 signal 1만 사용
    signal = record.__dict__['p_signal'][:,0]

    inds = annotation.__dict__['sample']
    symbols = annotation.__dict__['symbol']
    QT_inds_signal, signal = get_QT_labels(inds, symbols, signal)
    
    inds_to_keep = QT_inds_signal
    
    tmp_dict = {'record_name': record_name, 'signal': signal, 'qt_signal': QT_inds_signal,'inds_to_keep': inds_to_keep}
    labeled_data.append(tmp_dict)
    
labeled_data = pd.DataFrame(labeled_data)



#심전도 기저선 변동 잡음 제거를 위한 low pass filter 연산
N = 350
x = np.array(range(N))
det = np.sum(x**2)*N - np.sum(x)*np.sum(x)
A = np.array([[N, -np.sum(x)], [-np.sum(x), np.sum(x**2)]])

b = pd.read_csv('lpf.csv')
b = b['b'].values/(2**15)


def get_linear_fit(window):
    y = np.array([np.sum(window*x), np.sum(window)])
    m, b = (1/det)*np.matmul(A,y)
    linear_fit = m*x + b

    #return the midpoint
    return linear_fit[int(N/2)]



def remove_baseline(row):
    data = row['signal']
    record_name = row['record_name']

    if data is None:
        return row
    if len(data) < 500:
        return row

    baseline = np.ones(len(data))*np.nan
    for i in range(0, len(data)-N):
        center = get_linear_fit(data[i:i+N])
        baseline[int(i+N/2)] = center

    baseline_removed = data - baseline

    non_nan_inds = np.where(np.isfinite(baseline_removed))[0]
    baseline_removed = baseline_removed[non_nan_inds]
    row['signal'] = baseline_removed
    data = data[non_nan_inds]

    qt = row['qt_signal']
    qt = qt[np.isin(qt, non_nan_inds)]-non_nan_inds[0]
    row['qt'] = qt

    inds_to_keep = row['inds_to_keep']
    inds_to_keep = inds_to_keep[np.isin(inds_to_keep, non_nan_inds)]-non_nan_inds[0]
    row['inds_to_keep'] = inds_to_keep
    
   

    return row


df = labeled_data.apply(remove_baseline, axis=1)



#scalogram(연속 웨이블릿 변환) 모델의 계산 복잡성을 줄이기 위해 250hz에서 125hz로 다운샘플링
def get_scalogram(row):
    data = row['signal']
    record_name = row['record_name']

    if data is None:
        row['cwt'] = None
        return row

    if len(data) < 500:
        return row

    #filtfilt filter를 데이터와 맞춰줌
    data = filtfilt(b, 1, data)
    data = data[0::2]
    row['signal'] = data

    #morlet wavelet 사용
    wavelet = 'morl'
    scales = np.arange(2,64)  
    coefficients, frequencies = pywt.cwt(data=data, scales=scales, wavelet=wavelet, sampling_period=1/125)

    qt = row['qt']
    qt = np.unique((qt/2).astype(int))
    row['qt'] = qt

    inds_to_keep = row['inds_to_keep']
    inds_to_keep = np.unique((inds_to_keep/2).astype(int))
    row['inds_to_keep'] = inds_to_keep

    row['cwt'] = coefficients

    return row


df = df.apply(get_scalogram, axis=1)


# 데이터 생성 간에 환자 분리 진행(train 51명, valid 6명, test 6명)
train = df[:51]
valid = df[51:57]
test = df[57:]


# 데이터 크기는 125 간격은 10으로 설정
WIN_LEN = 125
WIN_SLIDE = 10

# 연속웨이블릿변환(morl 변환)
wavelet = 'morl'
scales = np.arange(2,64)
wavelet_freqs = pywt.scale2frequency(wavelet, scales)*125


def wavelet_trans(row):
    record_name = row['record_name']


    coefficients = row['cwt']
    
    if np.isnan(coefficients).all() ==True: return []
    data = row['signal']
    N = np.shape(coefficients)[1]
    QT_labels = np.zeros(N).astype(int)
    QT_labels[row['qt']] = 1
    valid_inds = row['inds_to_keep']

    windows = []

    for i in range(0, N-WIN_LEN, WIN_SLIDE):
        
        tmp_inds = range(i,i+WIN_LEN)
        tmp_win = coefficients[:,tmp_inds]
        tmp_labels = QT_labels[tmp_inds]
        tmp_data = data[tmp_inds]

        
        tmp_dict = {'window': tmp_win, 'label': tmp_labels, 'data': tmp_data}
        windows.append(tmp_dict)

    return windows


def get_dataset(patient):
    windows = patient.apply(wavelet_trans, axis=1)
    windows = pd.DataFrame(windows)
    windows = pd.DataFrame([item for sublist in windows.values for item in sublist[0]])
    windows = windows.reset_index(drop=True)
    
    return windows



train_windows = get_dataset(train)
print('train data: %i' % (len(train_windows)))

valid_windows = get_dataset(valid)
print('valid data: %i' % (len(valid_windows)))

test_windows = get_dataset(test)
print('test data: %i' % (len(test_windows)))

train_windows.to_pickle('train.pkl', protocol=4)
valid_windows.to_pickle('valid.pkl', protocol=4)
test_windows.to_pickle('test.pkl', protocol=4)

