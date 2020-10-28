import numpy as np
import csv
import random

def LoadData(file):
    with open(file, 'r') as fp:
        data_set = [i for i in csv.reader(fp)]
    return data_set


def Divide(data_set):
    train_set = random.sample(data_set, int(len(data_set)*0.8))
    test_set = [i for i in data_set if i not in train_set]
    return train_set, test_set


def Str2Float(list_in):
    return list(map(float, list_in))


def OneHot(data_set, index, attribute_set):
    # 特征值集合
    attribute_value = list(set([line[index] for line in data_set]))
    attribute_value = sorted(attribute_value)
    # 特征取值只有0与1则无需编码
    if(len(attribute_value) == 2):
        for i in range(0, len(data_set)):
            data_set[i] = data_set[i][0:index] + \
                data_set[i][index+1:] + [data_set[i][index]]
        attribute_set.append(attribute_set[index])
    # 特征取值唯一则删除
    elif(len(attribute_value) == 1):
        for i in range(0, len(data_set)):
            data_set[i] = data_set[i][0:index]+data_set[i][index+1:]
    # One-Hot
    else:
        # 将编码后的新特征放到数据集后
        for i in range(0, len(data_set)):
            one_hot = ['0' for temp in range(0, len(attribute_value))]
            # 对应特征位为1
            one_hot[attribute_value.index(data_set[i][index])] = '1'
            data_set[i] = data_set[i][0:index]+data_set[i][index+1:]+one_hot
        # 新特征名为 原特征名+特征取值
        for i in range(0, len(attribute_value)):
            attribute_set.append(attribute_set[index]+attribute_value[i])
    attribute_set.pop(index)
    return


def Normalization(data_set, index):
    X = [line[index] for line in data_set]
    # 平均值
    X_avg = np.mean(X, axis=0)
    # 方差
    X_std = np.std(X, axis=0)
    # 归一化
    for i in range(0, len(data_set)):
        data_set[i][index] -= X_avg
        data_set[i][index] /= X_std
    return


def DataPreProcessing(data_set):
    attribute_set = [i for i in data_set[0][1:]]
    data_set.pop(0)
    for i in range(0, len(data_set)):
        data_set[i] = data_set[i][1:]
    attribute_set.pop(0)
    # 日期处理
    for i in range(0, len(data_set)):
        date = data_set[i][0].split('/')[-1]
        data_set[i] = [date]+data_set[i][1:]
    attribute_set.insert(0, 'date')
    # One-Hot
    for i in range(0, 9):
        OneHot(data_set, 0, attribute_set)
    for i in range(0, len(data_set)):
        data_set[i] = Str2Float(data_set[i])
    # 归一化
    for i in range(0, 4):
        Normalization(data_set, i)
    train_set, test_set = Divide(data_set)
    # 训练集标签列表
    y_t = []
    for i in range(0, len(train_set)):
        y_t.append(train_set[i][4])
        train_set[i].pop(4)
    # 测试集标签列表
    y_e = []
    for i in range(0, len(test_set)):
        y_e.append(test_set[i][4])
        test_set[i].pop(4)
    return train_set, test_set, y_t, y_e


def BPNN(train_set, test_set, learning_rate, y_t, y_e, M, N, max_iter):
    # 对权重与偏置随机初始化
    w_MN = np.random.rand(M, N)
    b_M = np.random.rand(M)
    w_M = np.random.rand(M)
    b = random.random()

    count = 0
    x_in = np.array(train_set)
    y_r = np.array(y_t)
    # 开始迭代
    out = []
    while max_iter > count:
        # 初始化梯度
        delta_w_M = np.zeros((M))
        delta_b = 0.0
        delta_w_MN = np.zeros((M, N))
        delta_b_M = np.zeros(M)
        # 隐藏层输出
        O = (np.exp(2*(np.dot(x_in, w_MN.T)+b_M))-1)/(np.exp(2*(np.dot(x_in, w_MN.T)+b_M))+1)# tanh函数
        # O = 1/(1+np.exp(-np.dot(x_in, w_MN.T)-b_M)) # Sigmoid函数
        # O = np.maximum(0.01*(np.dot(x_in, w_MN.T)+b_M),np.dot(x_in, w_MN.T)+b_M)# LRelu函数
        # 输出层输出
        R = np.dot(O, w_M)+b
        # 误差
        E = 1/2*(R-y_r)**2
        # 计算隐藏层到输出层权重的梯度
        delta_w_M = learning_rate*O*(R-y_r).reshape((-1, 1))
        # 所有样例梯度累加
        delta_w_M = delta_w_M.sum(axis=0)
        # 计算隐藏层到输出层偏置的梯度
        delta_b = learning_rate*(R-y_r)
        # 所有样例梯度累加
        delta_b = delta_b.sum(axis=0)
        # 计算输入层到隐藏层偏置的梯度
        delta_b_M = learning_rate*(R-y_r).reshape((-1, 1))*w_M*(1-O*O)
        # delta_b_M = learning_rate*(R-y_r).reshape((-1, 1))*w_M*np.minimum(np.maximum(O,0.01),1)# LRelu函数
        # 计算输入层到隐藏层权重的梯度
        delta_w_MN = np.dot(delta_b_M.T, x_in)
        delta_b_M = delta_b_M.sum(axis=0)

        # 更新权重与偏置
        w_MN = w_MN-delta_w_MN/len(train_set)
        w_M = w_M-delta_w_M/len(train_set)
        b = b-delta_b/len(train_set)
        b_M = b_M-delta_b_M/len(train_set)
        count += 1
        # 每次迭代结束后的均方误差
        print(count)
        print(E.sum()/len(train_set))
        out.append(E.sum()/len(train_set))


    temp_in = np.array(test_set)
    y_temp = np.array(y_e)

    O = (np.exp(2*(np.dot(temp_in, w_MN.T)+b_M))-1)/(np.exp(2*(np.dot(temp_in, w_MN.T)+b_M))+1)# tanh函数
    # O = np.maximum(0.01*(np.dot(temp_in, w_MN.T)+b_M),np.dot(temp_in, w_MN.T)+b_M)# LRelu函数
    # O = 1/(1+np.exp(-np.dot(temp_in, w_MN.T)-b_M))# Sigmoid函数
    # 输出层输出
    R = np.dot(O, w_M)+b
        # 误差
    E = 1/2*(R-y_temp)**2
    print("测试集误差")
    print(E.sum()/len(test_set))
    return


if __name__ == "__main__":
    file = 'lab4_dataset/train.csv'
    data_set = LoadData(file)
    train_set, test_set, y_t, y_e = DataPreProcessing(data_set)
    # 隐藏层神经元数
    M = 65
    # 输入层神经元数
    N = len(data_set[0])
    # 学习率
    learning_rate = 0.01
    BPNN(train_set, test_set, learning_rate, y_t, y_e, M, N, 1000)
