import csv
import numpy as np
import math

#读取训练集建立词库
def LoadWordSet():
    csv_path = 'lab1_data/regression_dataset/train_set.csv'
    with open(csv_path, 'r') as fp:
        train_set = [i for i in csv.reader(fp)]
    word_set = set()
    for i in range(1, len(train_set)):
        for word in train_set[i][0].split():
            word_set.add(word)
    word_set = sorted(word_set)
    return word_set, train_set

#计算tf-idf矩阵
def tf_idf(word_set, sentence):
    count_dict = dict()
    for word in word_set:
        count_dict[word] = 0
    word_tf = []
    for words in sentence:
        word_count = []
        for word in word_set:
            if len(words) == 0:
                word_count.append(words.count(word))
            else:
                word_count.append((words.count(word))/len(words))
            if words.count(word) > 0:
                if word in count_dict:
                    count_dict[word] += 1
        word_tf.append(word_count)
    for word in count_dict:
        count_dict[word] = math.log(len(sentence)/(count_dict[word]+1))

    for document in word_tf:
        for index in range(len(document)):
            document[index] = document[index] * count_dict[word_set[index]]
    return word_tf

#计算训练集的tf-idf矩阵
def TFIDF_TrSet(word_set, train_set):
    KNN_TR = []
    sentence = [train_set[i][0].split() for i in range(1, len(train_set))]
    Tr_tf = tf_idf(word_set, sentence)
    for i in range(1, len(train_set)):
        KNN_TR.append([Tr_tf[i-1], list(map(float, train_set[i][1:7]))])
    return KNN_TR

#计算验证集的tf-idf矩阵
def TFIDF_VaSet(word_set):
    csv_path = 'lab1_data/regression_dataset/validation_set.csv'
    with open(csv_path, 'r') as fp:
        validation_set = [i for i in csv.reader(fp)]
    KNN_VA = []
    sentence = [validation_set[i][0].split()
                for i in range(1, len(validation_set))]
    Tr_tf = tf_idf(word_set, sentence)
    for i in range(1, len(validation_set)):
        KNN_VA.append([Tr_tf[i-1], list(map(float, validation_set[i][1:7]))])
    return KNN_VA

#循环查找最好的K值
def FindBestK(KNN_TR, KNN_VA):
    P = 1
    K = int(20)
    #采用Lp距离计算验证集各例与验证集的距离
    distance_set = []
    for VA in KNN_VA:
        distance = []
        for TR in KNN_TR:
            #距离公式
            SUM = np.sum(
                np.power(np.abs(np.array([TR[0]])-np.array([VA[0]])), P))
            distance.append(np.power(SUM, 1/P))
        distance_set.append(distance)
    #筛选出距离最小的K个例子
    Result = []
    for d in distance_set:
        d_sort = np.sort(d)
        index_sort = np.argsort(d)
        R_K = []
        for j in range(1, K):
            #筛选出距离最小的K个例子对应的标签概率并除以距离
            top_k = [(np.array(KNN_TR[i][1])/(d_sort[i])).tolist()
                     for i in index_sort[:j]]
            Sum_k = np.array(top_k[0])
            #根据回归公式对得到的top_K个标签概率进行求和
            for n in range(1, len(top_k)):
                Sum_k += np.array(top_k[n])
            #对得到的结果进行归一化
            temp = np.sum(Sum_k)
            for pi in range(0, len(Sum_k)):
                Sum_k[pi] /= temp
            R_K.append(Sum_k.tolist())
        Result.append(R_K)
    #计算每个K值对应的相关系数
    corr_factor = []
    for i in range(0, K-1):
        corr_k = []
        #计算6个标签的相关系数
        for j in range(0, 6):
            alist = [row[i][j] for row in Result]
            blist = [row[1][j] for row in KNN_VA]
            #平均值
            a_avg = sum(alist)/len(alist)
            b_avg = sum(blist)/len(blist)
            #协方差
            cov_ab = np.sum(np.multiply(np.array(alist)-a_avg,np.array(blist)-b_avg))
            #方差
            d_a = math.sqrt(np.sum(np.power(np.array(alist)-a_avg,2)))
            d_b = math.sqrt(np.sum(np.power(np.array(blist)-b_avg,2)))
            #协方差与方差之比
            corr_k.append(cov_ab/(d_a*d_b))
        #最后结果取平均
        corr_factor.append(sum(corr_k)/len(corr_k))
    #对准确度进行排序后输出效果最好的K
    K_best = np.argsort(corr_factor)[-1]+1
    print('K    相关系数')
    for i in range(len(corr_factor)):
        print("%-5d%f" %(i+1,corr_factor[i]))
    return K_best

#读取测试集并计算tf-idf矩阵
def TFIDF_TeSet(word_set):
    csv_path = 'lab1_data/regression_dataset/test_set.csv'
    with open(csv_path, 'r') as fp:
        test_set = [i for i in csv.reader(fp)]
    KNN_TE = []
    sentence = [test_set[i][1].split() for i in range(1, len(test_set))]
    Tr_tf = tf_idf(word_set, sentence)
    for i in range(1, len(test_set)):
        KNN_TE.append([Tr_tf[i-1], []])
    return KNN_TE, test_set

#使用最好的K值进行KNN算法
def KNN(BestK, KNN_TR, KNN_TE, test_set):
    P = 1
    #计算距离，同上
    distance_set = []
    for TE in KNN_TE:
        distance = []
        for TR in KNN_TR:
            SUM = np.sum(
                np.power(np.abs(np.array([TR[0]])-np.array([TE[0]])), P))
            distance.append(np.power(SUM, 1/P))
        distance_set.append(distance)
    #输出结果   
    Regression = []
    for d in distance_set:
        d_sort = np.sort(d)
        index_sort = np.argsort(d)
        top_k = [(np.array(KNN_TR[i][1])/(d_sort[i])).tolist()
                 for i in index_sort[:BestK]]
        Sum_k = np.array(top_k[0])
        for n in range(1, len(top_k)):
            Sum_k += np.array(top_k[n])
        temp = np.sum(Sum_k)
        for pi in range(0, len(Sum_k)):
            Sum_k[pi] /= temp
        Regression.append(Sum_k.tolist())
    out = []
    for i in range(1, len(test_set)):
        out_line = [test_set[i][1]]+list(map(str, Regression[i-1]))
        out.append(out_line)
    return out


if __name__ == '__main__':
    word_set, train_set = LoadWordSet()
    KNN_TR = TFIDF_TrSet(word_set, train_set)
    KNN_VA = TFIDF_VaSet(word_set)
    BestK = FindBestK(KNN_TR, KNN_VA)
    print("最好的K值为%d" % BestK)
    KNN_TE, test_set = TFIDF_TeSet(word_set)
    Result = KNN(BestK, KNN_TR, KNN_TE, test_set)
    with open('18340236_ZhuYu_KNN_regression.csv', 'w', newline='') as fp:
        write = csv.writer(fp)
        write.writerows(Result)
