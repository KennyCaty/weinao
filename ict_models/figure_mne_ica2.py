import pandas as pd
import mne
import scipy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 后台绘图
from scipy import signal
from sklearn.decomposition import FastICA
import os
import uuid
from datetime import datetime

def create_unique_folder(save_dir):
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-8]

    # 生成一个随机的UUID
    unique_id = str(uuid.uuid4())

    # 结合时间戳和UUID创建文件夹名称
    folder_name = f"{timestamp}_{unique_id}"

    # 定义文件夹路径，可以根据需要修改路径
    folder_path = os.path.join(f"{save_dir}/", folder_name)

    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)

    return folder_path





def mne_picture(data_path, save_dir):
    # 调用函数创建唯一文件夹
    new_folder = create_unique_folder(save_dir)
    # print(f"Created folder: {new_folder}")
    

    # 读取Excel文件
    # df = pd.read_excel('C:\pycharmProject\qt\qt2\mne_data\\4.23_22.26.47-4.23_22.34.52.xls')
    df = pd.read_csv(data_path, sep=',')
    # 提取36列数据

    # data = np.transpose(data)
    df0 = df[df.iloc[:, 36] == 0]
    df1 = df[df.iloc[:, 36] == 1]
    df2 = df[df.iloc[:, 36] == 2]
    df3 = df[df.iloc[:, 36] == 3]
    df4 = df[df.iloc[:, 36] == 4]
    df = np.concatenate((df0, df1, df2, df3, df4))
    df = pd.DataFrame(df)


    ch_names_data = ['T5', 'FT7', 'P3', 'FC3', 'Pz', 'FCz', 'P4', 'FC4', 'C4', 'Fp2', 'T4', 'Fp1', 'TP7', 'F7', 'Cz', 'O2', 'C3', 'Oz', 'T3', 'O1','FT8', 'T6', 'F8', 'TP8', 'F4', 'CP4', 'Fz', 'CPz', 'F3', 'CP3']
    data = df.iloc[4000:6200, [0,  1,  2,  3,  4,  5,  6,  7,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
    data = np.transpose(data)


    ch_types_data = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg',\
                     'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']  # 通道类型

    sfreq = 250
    projs=True

    info_data = mne.create_info(ch_names_data, sfreq, ch_types_data)  # 创建信号的信息
    info_data.set_montage('standard_1020')
    raw_data = mne.io.RawArray(data, info_data)

    raw_data_alpha = raw_data.filter(l_freq=0.5, h_freq=100)
    raw_data_alpha = raw_data_alpha.notch_filter(freqs=50)  # 去除交流电影响

    pic0 = raw_data_alpha.plot( block=True,scalings='auto')
    pic0.savefig(f'{new_folder}/0.png')  # 可自定义保存的文件名和格式
    # plt.show()
    # pic.savefig('mne_photo/1.png')  # 可自定义保存的文件名和格式
    pic1 = raw_data_alpha.plot_psd(fmin=0, fmax=100)
    pic1.savefig(f'{new_folder}/1.png')  # 可自定义保存的文件名和格式


    ica = mne.preprocessing.ICA(n_components=30, method='fastica', max_iter=800)
    ica.fit(raw_data_alpha)
    pic2 = ica.plot_sources(raw_data_alpha)
    pic2.savefig(f'{new_folder}/2.png')  # 可自定义保存的文件名和格式
    pic3 = ica.plot_components()  # 可视化每个独立成分的头皮分布
    pic3[0].savefig(f'{new_folder}/3.png') # 可自定义保存的文件名和格式
    # 绘制每个独立成分的一些属性
    pick_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    # 保存路径
    prop_save_dir = f'{new_folder}/properties/'
    os.makedirs(prop_save_dir, exist_ok=True)

    for i, pick in enumerate(pick_id):
        prop_fig = ica.plot_properties(raw_data_alpha, picks=[pick], show=False)
        prop_fig[0].savefig(os.path.join(prop_save_dir, f'property_{i}.png'))

    # pics = ica.plot_properties(raw_data_alpha, picks=pick_id)
    # plt.savefig('mne_photo/5.png')
        
    return new_folder  # 返回保存的文件夹路径

if __name__ == '__main__':
    data_path = "mne_data/王家乐_02.csv"
    mne_picture(data_path, 'mne_photos')


