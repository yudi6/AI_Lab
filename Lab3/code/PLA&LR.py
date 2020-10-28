import numpy
import csv
import random
import math


def LoadData(file):
    with open(file, 'r') as fp:
        data_set = [i for i in csv.reader(fp)]
    return data_set


def Divide(data_set, proportion):
    # 计算原数据集中正反例的个数
    label_dict = {'0': 0, '1': 0}
    for i in range(1, len(data_set)):
        if data_set[i][-1] == '0':
            label_dict['0'] += 1
        else:
            label_dict['1'] += 1
    print(label_dict)
    # 根据比例得到验证集中正反例应该有的数目
    for value in label_dict:
        label_dict[value] = int(label_dict[value]*proportion)
    select_set = [data_set[i] for i in range(1, len(data_set))]

    train_set = []
    test_set = []
    while len(select_set) > 0:
        # 生成随机数，随机抽样
        i = random.randint(0, len(select_set)-1)
        # 进行分层抽样
        if select_set[i][-1] == '0':
            if label_dict['0'] > 0:
                train_set.append(select_set[i])
                label_dict['0'] -= 1
                select_set.pop(i)
            else:
                test_set.append(select_set[i])
                select_set.pop(i)
        else:
            if label_dict['1'] > 0:
                train_set.append(select_set[i])
                label_dict['1'] -= 1
                select_set.pop(i)
            else:
                test_set.append(select_set[i])
                select_set.pop(i)
    return train_set, test_set


def Str2Float(list_in):
    return list(map(float, list_in))


def PLA(train_set, w, b, n, max_i, test_set):
    print("迭代次数  正确率")
    # 判断是否存在误分类点
    test = True
    AC_set = []
    best_w = [i for i in w]
    best_b = b
    while test == True and max_i > 0:
        test = False
        for line in train_set:
            x_i = line[:-1]
            x_i = Str2Float(x_i)
            y_i = 1*(line[-1] == '1') + (-1)*(line[-1] == '0')
            # 存在误分类点，对w进行更新
            if y_i*(numpy.matmul(w, x_i)+b) <= 0:
                # w = w + n*yi*xi
                w = list(numpy.array(w) + n*y_i*numpy.array(x_i))
                b = b + n*y_i
                test = True
                # 迭代次数减少
                max_i -= 1
                AC = PLA_AC(test_set, w, b)
                AC_set.append(AC)
                # 保存最好的w与b
                if AC >= max(AC_set):
                    best_w = [i for i in w]
                    best_b = b
                print("%-10d%f" % (500-max_i, AC))
                break
    print("PLA最好迭代次数  最好正确率")
    print("%-15d%f" % (AC_set.index(max(AC_set))+1, max(AC_set)))            
    return best_w, best_b


def LR(train_set, w, n, max_i, test_set):
    print("迭代次数  正确率")
    while max_i > 0:
        # 批梯度
        lw = numpy.array([0.0 for i in range(0, len(w))])
        for line in train_set:
            x_i = line[:-1]
            x_i = Str2Float(x_i)
            x_i.append(1.0)
            y_i = 1*(line[-1] == '1') + (0)*(line[-1] == '0')
            # 防止数据溢出 对梯度进行求和
            if -numpy.matmul(x_i, w) > 200:
                lw += n*(-y_i)*numpy.array(x_i)
            else:
                lw += n*((1/(1+math.exp(-numpy.matmul(x_i, w)))-y_i)
                         * numpy.array(x_i))
        # 更新w
        w = list(numpy.array(w) - lw)
        max_i -= 1
        print("%-10d%f" % (500-max_i, LR_AC(test_set, w)))
        # 根据范数判断是否收敛
        if numpy.linalg.norm(lw, ord=2) < 0.001:
            break
    return w


def PLA_AC(test_set, w, b):
    count = 0
    for line in test_set:
        x_r = line[:-1]
        x_r = Str2Float(x_r)
        y_r = 1*(line[-1] == '1') + (-1)*(line[-1] == '0')
        # 判断分类是否正确
        if y_r*(numpy.matmul(w, x_r)+b) > 0:
            count += 1
    return count/len(test_set)


def LR_AC(test_set, w):
    count = 0
    for line in test_set:
        x_r = line[:-1]
        x_r = Str2Float(x_r)
        x_r.append(1.0)
        y_r = 1*(line[-1] == '1') + (0)*(line[-1] == '0')
        # 防止数据溢出
        if -numpy.matmul(w, x_r) > 200:
            pi_x = 0
        else:
            pi_x = 1/(1 + math.exp(-numpy.matmul(w, x_r)))
        # 判断分类是否正确
        if (pi_x >= 0.5 and y_r == 1) or (pi_x < 0.5 and y_r == 0):
            count += 1
    return count/len(test_set)


if __name__ == "__main__":
    file = 'train.csv'
    data_set = LoadData(file)
    proportion = 0.8
    train_set, test_set = Divide(data_set, proportion)

    w = [0.0 for it in range(len(train_set[0])-1)]
    b = 0
    n = 1
    w_r, b_r = PLA(train_set, w, b, n, 500, test_set)
    #print(w_r,b_r)
    #AC = PLA_AC(test_set, w_r, b_r)
    #print(AC)

    w = [0.0 for it in range(len(train_set[0])-1)]
    n = 0.00001
    w.append(0.0) 
    w_r = LR(train_set, w, n, 500, test_set)
