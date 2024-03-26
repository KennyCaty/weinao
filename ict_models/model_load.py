import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Flatten
from keras.layers import LayerNormalization, MultiHeadAttention
from keras.layers import Bidirectional, GRU
from sklearn.decomposition import FastICA
from keras.layers import LayerNormalization
from keras.optimizers import Adam

from scipy import signal
def main_model_load(data_path):
    # 定义注意力层
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="normal", trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            e = tf.keras.backend.dot(x, self.W)
            e = tf.keras.backend.tanh(e)
            a = tf.keras.backend.softmax(e, axis=1)
            output = x * a
            return tf.keras.backend.sum(output, axis=1)

    # 读取第一个CSV文件
    data1 = pd.read_csv(f'{data_path}', sep=',')
    # 去除最后一行数据
    df1 = data1.iloc[:-1]
    # 定义一个空列表用于存储分割后的样本数据
    samples = []
    sample = []
    index_n = 1
    # 遍历dataframe，根据条件分割样本数据
    for index, row in df1.iterrows():
        curr_value = row[36]  # 第37列的值

        if curr_value == 4:
            next_row = data1.iloc[index + 1]  # 获取下一行数据
            next_value = next_row[36]  # 下一行的第37列的值
            if next_value == 0:
                if sample:
                    samples.append(pd.DataFrame(sample))  # 存储上一个样本数据
                sample = []  # 开始一个新的样本数据
        sample.append(row)  # 添加行数据到样本数据中
    # 最后一个样本数据需要单独处理
    samples.append(pd.DataFrame(sample))

    # 划分训练集和测试集
    train_data1 = data1.sample(frac=0.8, random_state=42)
    test_data1 = data1.drop(train_data1.index)

    # 定义输入数据和输出标签
    train_X1 = train_data1.iloc[1:, :36].values
    train_Y1 = [train_data1.iloc[1:, 38].values, train_data1.iloc[1:, 39].values, train_data1.iloc[1:, 40].values,
                train_data1.iloc[1:, 41].values, train_data1.iloc[1:, 42].values, train_data1.iloc[1:, 43].values,
                train_data1.iloc[1:, 44].values]
    train_f1_1 = train_data1.iloc[1:, 36].values
    train_f2_1 = train_data1.iloc[1:, 45].values

    test_X1 = test_data1.iloc[1:, :36].values
    test_Y1 = [test_data1.iloc[1:, 38].values, test_data1.iloc[1:, 39].values, test_data1.iloc[1:, 40].values,
               test_data1.iloc[1:, 41].values, test_data1.iloc[1:, 42].values, test_data1.iloc[1:, 43].values,
               test_data1.iloc[1:, 44].values]
    test_f1_1 = test_data1.iloc[1:, 36].values
    test_f2_1 = test_data1.iloc[1:, 45].values

   
    def convert_labels(y):
        unique_vals = np.unique(y)
        if len(unique_vals) == 1:
            return np.zeros_like(y)
        else:
            min_val, max_val = sorted(unique_vals)
            return np.where(y == max_val, 1, 0)

    # Convert the labels for training and testing data
    # train_Y_final = [convert_labels(y) for y in train_Y_combined]
    train_Y_final = [convert_labels(y) for y in train_Y1]
    test_Y_final = [convert_labels(y) for y in test_Y1]
    # test_Y_final = [convert_labels(y) for y in test_Y_final]

    # 数据预处理
    def preprocess_data(X):
        # 带通滤波
        fs = 250  # 采样频率
        lowcut = 1  # 低频截止频率
        highcut = 50  # 高频截止频率
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        X_filtered = signal.filtfilt(b, a, X, axis=0)
        # ICA去伪迹
        ica = FastICA(n_components=36, tol=1e-3, max_iter=1500)
        X_processed = ica.fit_transform(X_filtered)
        return X_processed
    X_processed_train = preprocess_data(train_X1)
    X_processed_test = preprocess_data(test_X1)

    # train_Y = tf.keras.utils.to_categorical(train_Y)  # 将标签转换为 one-hot 编码
    # test_Y = tf.keras.utils.to_categorical(test_Y)
    X_processed_1_train = X_processed_train[:, [1,3,5,7,12,13,14,17,18,20,21,22,24,27,31]]
    X_processed_2_train = X_processed_train[:, [0,2,4,6,12,16,20,23]]
    X_processed_3_train = X_processed_train[:, [1,3,5,7,12,13,14,17,18,20,21,22,24,27,31]]
    X_processed_4_train = X_processed_train[:, [11,13,17,19,21]]
    X_processed_5_train = X_processed_train[:, [10,11,12,13,16,18,20]]
    X_processed_6_train = X_processed_train[:, [11,12,13,16,20]]
    X_processed_7_train = X_processed_train[:, [17,19,21]]

    X_processed_1_test = X_processed_test[:, [1,3,5,7,12,13,14,17,18,20,21,22,24,27,31]]
    X_processed_2_test = X_processed_test[:, [0,2,4,6,12,16,20,23]]
    X_processed_3_test = X_processed_test[:, [1,3,5,7,12,13,14,17,18,20,21,22,24,27,31]]
    X_processed_4_test = X_processed_test[:, [11,13,17,19,21]]
    X_processed_5_test = X_processed_test[:, [10,11,12,13,16,18,20]]
    X_processed_6_test = X_processed_test[:, [11,12,13,16,20]]
    X_processed_7_test = X_processed_test[:, [17,19,21]]



    # 输入层
    input1 = Input(shape=(15, 1))
    input2 = Input(shape=(8, 1))
    input3 = Input(shape=(15, 1))
    input4 = Input(shape=(5, 1))
    input5 = Input(shape=(7, 1))
    input6 = Input(shape=(5, 1))
    input7 = Input(shape=(3, 1))
    input_features_1 = Input(shape=(1,))
    input_features_2 = Input(shape=(1,))

    # 共享的BiGRU层
    shared_layer = Bidirectional(GRU(64, return_sequences=True))

    # 在输入上应用共享层
    output1 = shared_layer(input1)
    output2 = shared_layer(input2)
    output3 = shared_layer(input3)
    output4 = shared_layer(input4)
    output5 = shared_layer(input5)
    output6 = shared_layer(input6)
    output7 = shared_layer(input7)

    # 注意力层
    attention1 = AttentionLayer()(output1)
    attention2 = AttentionLayer()(output2)
    attention3 = AttentionLayer()(output3)
    attention4 = AttentionLayer()(output4)
    attention5 = AttentionLayer()(output5)
    attention6 = AttentionLayer()(output6)
    attention7 = AttentionLayer()(output7)

    # 将注意力输出与特征连接
    merged_1 = Concatenate()([attention1, input_features_1, input_features_2])
    merged_2 = Concatenate()([attention2, input_features_1, input_features_2])
    merged_3 = Concatenate()([attention3, input_features_1, input_features_2])
    merged_4 = Concatenate()([attention4, input_features_1, input_features_2])
    merged_5 = Concatenate()([attention5, input_features_1, input_features_2])
    merged_6 = Concatenate()([attention6, input_features_1, input_features_2])
    merged_7 = Concatenate()([attention7, input_features_1, input_features_2])

    # 添加全连接层
    dense1 = Dense(64, activation='relu')(merged_1)
    dense2 = Dense(64, activation='relu')(merged_2)
    dense3 = Dense(64, activation='relu')(merged_3)
    dense4 = Dense(64, activation='relu')(merged_4)
    dense5 = Dense(64, activation='relu')(merged_5)
    dense6 = Dense(64, activation='relu')(merged_6)
    dense7 = Dense(64, activation='relu')(merged_7)

    # 输出层
    output1 = Dense(6, activation='softmax')(dense1)
    output2 = Dense(9, activation='softmax')(dense2)
    output3 = Dense(8, activation='softmax')(dense3)
    output4 = Dense(9, activation='softmax')(dense4)
    output5 = Dense(8, activation='softmax')(dense5)
    output6 = Dense(9, activation='softmax')(dense6)
    output7 = Dense(6, activation='softmax')(dense7)

    # 定义Dense Block
    def dense_block(x, filters):
        x = tf.expand_dims(x, axis=-1)  # Adding the extra dimension
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Conv1D(filters, 3, padding='same')(x1)
        return tf.keras.layers.concatenate([x, x1])
    # 定义Transition Block
    def transition_block(x, filters):
        x = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
        return tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)

    # 定义第二个Dense Block
    def dense_block_2(x, filters):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Conv1D(filters, 3, padding='same')(x1)

        x2 = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.LeakyReLU()(x2)
        x2 = tf.keras.layers.Conv1D(filters, 3, padding='same')(x2)

        return tf.keras.layers.concatenate([x, x1, x2])

    # 定义Transition Block 2
    def transition_block_2(x, filters):
        x = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
        return tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)

    # 定义Dense Block 3
    def dense_block_3(x, filters):
        x1 = tf.keras.layers.BatchNormalization()(x)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Conv1D(filters, 3, padding='same')(x1)
        return tf.keras.layers.concatenate([x, x1])

    # 将输出添加到Dense Block
    dense1 = dense_block(output1, 64)
    dense2 = dense_block(output2, 64)
    dense3 = dense_block(output3, 64)
    dense4 = dense_block(output4, 64)
    dense5 = dense_block(output5, 64)
    dense6 = dense_block(output6, 64)
    dense7 = dense_block(output7, 64)

    # 在Dense Block输出上应用Transition Block
    transition1 = transition_block(dense1, 32)
    transition2 = transition_block(dense2, 32)
    transition3 = transition_block(dense3, 32)
    transition4 = transition_block(dense4, 32)
    transition5 = transition_block(dense5, 32)
    transition6 = transition_block(dense6, 32)
    transition7 = transition_block(dense7, 32)


    # 在Transition Block输出上应用第二个Dense Block
    dense1_2 = dense_block_2(transition1, 64)
    dense2_2 = dense_block_2(transition2, 64)
    dense3_2 = dense_block_2(transition3, 64)
    dense4_2 = dense_block_2(transition4, 64)
    dense5_2 = dense_block_2(transition5, 64)
    dense6_2 = dense_block_2(transition6, 64)
    dense7_2 = dense_block_2(transition7, 64)

    # 在第二个Dense Block的输出上应用Transition Block 2
    transition1_2 = transition_block_2(dense1_2, 32)
    transition2_2 = transition_block_2(dense2_2, 32)
    transition3_2 = transition_block_2(dense3_2, 32)
    transition4_2 = transition_block_2(dense4_2, 32)
    transition5_2 = transition_block_2(dense5_2, 32)
    transition6_2 = transition_block_2(dense6_2, 32)
    transition7_2 = transition_block_2(dense7_2, 32)

    # 在Transition Block 2的输出上应用Dense Block 3
    dense1_3 = dense_block_3(transition1_2, 64)
    dense2_3 = dense_block_3(transition2_2, 64)
    dense3_3 = dense_block_3(transition3_2, 64)
    dense4_3 = dense_block_3(transition4_2, 64)
    dense5_3 = dense_block_3(transition5_2, 64)
    dense6_3 = dense_block_3(transition6_2, 64)
    dense7_3 = dense_block_3(transition7_2, 64)

    # Flatten Transition Block 2输出
    flat1_2 = Flatten()(dense1_3)
    flat2_2 = Flatten()(dense2_3)
    flat3_2 = Flatten()(dense3_3)
    flat4_2 = Flatten()(dense4_3)
    flat5_2 = Flatten()(dense5_3)
    flat6_2 = Flatten()(dense6_3)
    flat7_2 = Flatten()(dense7_3)


    # 添加第二个全连接层
    dense1_2 = Dense(64, activation='relu')(flat1_2)
    dense2_2 = Dense(64, activation='relu')(flat2_2)
    dense3_2 = Dense(64, activation='relu')(flat3_2)
    dense4_2 = Dense(64, activation='relu')(flat4_2)
    dense5_2 = Dense(64, activation='relu')(flat5_2)
    dense6_2 = Dense(64, activation='relu')(flat6_2)
    dense7_2 = Dense(64, activation='relu')(flat7_2)

    # 添加第二个分类器
    output_1 = Dense(2, activation='softmax')(dense1_2)
    output_2 = Dense(2, activation='softmax')(dense2_2)
    output_3 = Dense(2, activation='softmax')(dense3_2)
    output_4 = Dense(2, activation='softmax')(dense4_2)
    output_5 = Dense(2, activation='softmax')(dense5_2)
    output_6 = Dense(2, activation='softmax')(dense6_2)
    output_7 = Dense(2, activation='softmax')(dense7_2)

    # 重新定义和编译模型
    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7, input_features_1, input_features_2], \
                  outputs=[output_1, output_2, output_3, output_4, output_5, output_6, output_7])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 载入训练好的模型权重
    model.load_weights(r'C:\Users\19396\Desktop\ict-web\ict_models\MCI_updated_model_weights1.h5')

    # 在测试数据上进行预测
    # predictions = model.predict([X_processed_1_test, X_processed_2_test, X_processed_3_test,
    #                               X_processed_4_test, X_processed_5_test, X_processed_6_test,
    #                               X_processed_7_test, test_f1_combined, test_f2_combined])

    predictions = model.predict([X_processed_1_test, X_processed_2_test, X_processed_3_test,
                                  X_processed_4_test, X_processed_5_test, X_processed_6_test,
                                  X_processed_7_test, test_f1_1, test_f2_1])

    # 你现在可以按需要使用预测结果，例如，打印每个输出的预测标签
    predict_result = []
    for i in range(7):
        predicted_labels = np.argmax(predictions[i], axis=1)
        predict_result.append(int(8-predicted_labels[0]))
        # print(f'输出{i+1}的预测标签：{predicted_labels}')
    return predict_result
if __name__ == '__main__':
    data_path = r'C:\Users\19396\Desktop\ict-web\data\wjl_02.csv'
    predict_result = main_model_load(data_path)
    print(predict_result)

    # data1 = pd.read_csv(r'C:\Users\19396\Desktop\ict-web\data\王家乐_02.csv', sep=',')