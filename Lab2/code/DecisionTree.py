import csv

import math

import random

from graphviz import Digraph


class DecisionTreeNode:
    def __init__(self, leaf=False, value=None, attribute=None, child_node=None):
        self.leaf = leaf  # 是否为叶节点
        self.value = value  # 叶节点则为最终取值
        self.attribute = attribute  # 中间节点则为划分特征
        self.child_node = child_node  # 子节点


def LoadData(file_path):
    with open(file_path, 'r') as fp:
        train_set = [i for i in csv.reader(fp)]
    return train_set


def Combinations(value_set):
    # 生成value_set的全排列，除去全空与其本身
    combinations = [[]]
    for value in value_set:
        combinations += [subset + [value] for subset in combinations]
    return combinations[1:len(combinations)-1]


def CalGiniByVal(data_set, attribute_index, value_sub_set):
    # 根据划分得到四种样例的数目：划分内正例，划分内反例，划分外正例，划分外反例
    gini_p = [[0, 0], [0, 0]]
    for i in range(len(data_set)):
        if data_set[i][attribute_index] in value_sub_set:
            if data_set[i][-1] == '0':
                gini_p[0][0] += 1
            else:
                gini_p[0][1] += 1
        else:
            if data_set[i][0] == '0':
                gini_p[1][0] += 1
            else:
                gini_p[1][1] += 1
    # print(gini_p)
    # 根据公式计算划分得到的GINI系数
    gini_rela = [1-(gini_p[0][0]/(gini_p[0][0]+gini_p[0][1]+0.000000001)) **
                 2-(gini_p[0][1]/(gini_p[0][0]+gini_p[0][1]+0.000000001))**2]
    gini_rela += [1-(gini_p[1][0]/(gini_p[1][0]+gini_p[1][1]+0.000000001))
                  ** 2-(gini_p[1][1]/(gini_p[1][0]+gini_p[1][1]+0.000000001))**2]
    # 计算在特征A情况下，数据集的GINI系数
    gini_rela[0] *= sum(gini_p[0])/(sum(gini_p[0])+sum(gini_p[1]))
    gini_rela[1] *= sum(gini_p[1])/(sum(gini_p[0])+sum(gini_p[1]))
    return sum(gini_rela)

# 计算信息熵


def CalEntropy(data_set, attribute_index=-1):
    # 计算下标对应的特征的熵，默认计算标签
    attribute_dict = dict()
    # 得到每个特征取值的数量
    for i in range(len(data_set)):
        attribute = data_set[i][attribute_index]
        if attribute in attribute_dict:
            attribute_dict[attribute] += 1
        else:
            attribute_dict[attribute] = 1
    # 信息熵
    Entropy = 0.0
    for attribute in attribute_dict:
        # 计算熵并求和
        Entropy += (- attribute_dict[attribute]/len(data_set)
                    * math.log2(attribute_dict[attribute]/len(data_set)))
    return Entropy


def CalRelativeEntropy(data_set, attribute_set, attribute_map):
    relative_entropy = []
    # 计算经验熵
    data_entropy = CalEntropy(data_set)
    for i in range(len(attribute_set)):
        attribute_value_set = attribute_map[attribute_set[i]]
        # print(attribute_set)
        attribute_entropy = 0.0
        sub_data_set = []
        for value in attribute_value_set:
            sub_data_set = SubDataSet(data_set, i, value)
            # print(len(sub_data_set[0]))
            # 计算每个特征下的条件熵
            attribute_entropy += (len(sub_data_set) /
                                  len(data_set))*CalEntropy(sub_data_set)
        # 计算信息增益
        relative_entropy.append(data_entropy - attribute_entropy)
    # 返回信息增益列表，与各特征对应
    return relative_entropy


def CalgRation(data_set, attribute_set):
    gRation = []
    # 计算每个特征的信息熵
    for i in range(len(attribute_set)):
        gRation.append(CalEntropy(data_set, i))
    # 计算每个特征对应的信息增益
    relative_entropy = CalRelativeEntropy(
        data_set, attribute_set, attribute_map)

    for i in range(len(gRation)):
        gRation[i] = relative_entropy[i] / (gRation[i] + 0.000000001)
    return gRation


def SubDataSet(data_set, attribute_index, attribute_value):
    # 返回只包含有传入特征值的数据集，并且删去该特征
    sub_data_set = []
    for i in range(len(data_set)):
        line = data_set[i]
        if line[attribute_index] == attribute_value:
            sub_data_set.append(
                line[:attribute_index]+line[attribute_index + 1:])
    return sub_data_set


def BestAttribute(data_set, attribute_set, function, attribute_map):
    if function == 'ID3':
        relative_entropy = CalRelativeEntropy(
            data_set, attribute_set, attribute_map)
        return relative_entropy.index(max(relative_entropy))

    elif function == 'C4.5':
        gRation = CalgRation(data_set, attribute_set)
        return gRation.index(max(gRation))


def FindBestMerge(data_set, attribute_set, attribute_index, attribute_map):
    # 根据GINI系数选择划分
    gini_merge = []
    attribute_combinations = Combinations(
        attribute_map[attribute_set[attribute_index]])

    for combination in attribute_combinations:
        gini_merge.append(CalGiniByVal(data_set, attribute_index, combination))
    combination_index = gini_merge.index(min(gini_merge))
    return [min(gini_merge), attribute_combinations[combination_index]]


def BestAttributeAndMerge(data_set, attribute_set, attribute_map):
    gini_merge = []
    for attribute_index in range(len(attribute_set)):
        gini_merge.append(FindBestMerge(
            data_set, attribute_set, attribute_index, attribute_map))
    best_index = 0
    for i in range(len(gini_merge)):
        if gini_merge[i][0] < gini_merge[best_index][0]:
            best_index = i
    return best_index, gini_merge[best_index][1]


def LabelIsSame(label_set):
    return len(set(label_set)) == 1


def VoteLabel(label_set):
    return max(label_set, key=label_set.count)


def MergeDataSet(data_set, merge, best_attribute):
    # 合并后的特征值字符串
    merge_str = str()
    # 除外的特征值字符串
    other_str = 'other'
    for i in merge:
        merge_str = merge_str + ' ' + i
    # print(merge_str)
    for line in data_set:
        if line[best_attribute] in merge:
            line[best_attribute] = merge_str
        else:
            line[best_attribute] = other_str
    return data_set, [merge_str, other_str]


def DecisionTree(data_set, attribute_set, function, attribute_map):
    # 所有样本的标签属于同个一类型，则不需要再划分，作为叶节点放回
    if LabelIsSame([line[-1] for line in data_set]):
        return DecisionTreeNode(True, data_set[0][-1])
    # 若所有特征都已经用于划分，则进行多数投票
    if len(attribute_set) == 0:
        return DecisionTreeNode(True, VoteLabel([line[-1] for line in data_set]))
    # CART算法
    if function == 'CART':
        # 最好的特征作为决策节点，以及相应特征值划分
        best_attribute, merge = BestAttributeAndMerge(
            data_set, attribute_set, attribute_map)
        #根据特征值划分新的数据集与特征取值[value, 'other']
        data_set, attribute_value_set = MergeDataSet(
            data_set, merge, best_attribute)
        # 新的特征集合
        new_attribute_set = [i for i in attribute_set]
        attribute_save = new_attribute_set.pop(best_attribute)
    # ID3算法和C4.5算法
    else:
        # 最好的特征作为决策节点，以及相应特征值划分
        best_attribute = BestAttribute(
            data_set, attribute_set, function, attribute_map)
        # 新的特征集合
        new_attribute_set = [i for i in attribute_set]
        attribute_save = new_attribute_set.pop(best_attribute)
        # 根据特征值划分新的数据集与特征取值
        attribute_value_set = attribute_map[attribute_save]
    # 根据特征取值获得子节点
    child_node = {}
    for value in attribute_value_set:
        # 划分子数据集
        sub_data_set = SubDataSet(data_set, best_attribute, value)
        # 子节点没有数据集则直接采用父节点的多数标签投票
        if len(sub_data_set) == 0:
            child_node[value] = DecisionTreeNode(
                True, VoteLabel([line[-1] for line in data_set]))
        else:
            # 递归建树
            child_node[value] = DecisionTree(
                sub_data_set, new_attribute_set, function, attribute_map)
    # 父节点指向子节点返回
    return DecisionTreeNode(attribute=attribute_save, child_node=child_node)


def PrintSubTree(graph, depth, node_name, decision_tree_node, key):
    if decision_tree_node.leaf == True:
        graph.node(name=node_name + key + str(depth),
                   label=str(decision_tree_node.value))
        graph.edge(node_name, node_name + key + str(depth), key)
    else:
        sub_node_name = node_name + key + \
            decision_tree_node.attribute + str(depth)
        graph.node(name=sub_node_name, label=str(decision_tree_node.attribute))
        graph.edge(node_name, sub_node_name, key)
        for sub_key in decision_tree_node.child_node:
            PrintSubTree(graph, depth+1, sub_node_name,
                         decision_tree_node.child_node[sub_key], sub_key)


def PrintDecisionTree(decision_tree_node, function):
    graph = Digraph(function + 'DecisionTree.gv', format='png')
    depth = 1
    node_name = decision_tree_node.attribute + str(depth)
    graph.node(name=node_name, label=decision_tree_node.attribute)
    for key in decision_tree_node.child_node:
        PrintSubTree(graph, depth + 1, node_name,
                     decision_tree_node.child_node[key], key)
    graph.render(function + 'DecisionTree', view=True)


def Classify(test_set, decision_tree, function, attribute_set):
    result = []
    # CART算法
    if function == 'CART':
        for line in test_set:
            # 从根节点进入树
            line_node = decision_tree
            while line_node.leaf == False:
                test = True
                index = attribute_set.index(line_node.attribute)
                # 根据特征取值进行判断进入子节点
                for key in line_node.child_node:
                    if line[index] in key.split():
                        line_node = line_node.child_node[key]
                        test = False
                if test == True:
                    line_node = line_node.child_node['other']
            # 到达叶节点则生成结果
            result.append(line_node.value)
    # ID3和C4.5
    else:
        for line in test_set:
            # 从根节点进入
            line_node = decision_tree
            while line_node.leaf != True:
                index = attribute_set.index(line_node.attribute)
                # 根据特征取值进入子节点
                for key in line_node.child_node:
                    if line[index] == key:
                        line_node = line_node.child_node[key]

            result.append(line_node.value)
    return result


def AC(true_result, result):
    count = 0
    for i in range(len(true_result)):
        if true_result[i] == result[i]:
            count += 1
    return count/len(true_result)


def Divide(data_set, proportion):
    # 计算原数据集中正反例的个数
    label_dict = {'0': 0, '1': 0}
    for i in range(1, len(data_set)):
        if data_set[i][-1] == '0':
            label_dict['0'] += 1
        else:
            label_dict['1'] += 1
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


if __name__ == "__main__":
    # 训练集占比
    proportion = 0.7
    result_list = []
    # 生成树的数目，n*3
    n = 8
    #AC_list = []
    file_path = "car_train.csv"
    data_set = LoadData(file_path)
    org_train_set, org_test_set = Divide(data_set, 0.1)
    true_result = [i[-1] for i in org_test_set]
    while n > 0:
        file_path = "car_train.csv"
        data_set = LoadData(file_path)
        print("此时的p为:%f" % proportion)
        attribute_map = {}
        attribute_set = data_set[0][:6]
        for i in range(len(attribute_set)):
            attribute_value_set = list(
                set([data_set[j][i] for j in range(1, len(data_set))]))
            attribute_map[attribute_set[i]] = sorted(attribute_value_set)

        org_train_set, test_set = Divide(data_set, proportion)
        train_set = [i for i in org_train_set]
        new_attribute_set = [i for i in attribute_set]

        decision_tree = DecisionTree(
            train_set, new_attribute_set, 'ID3', attribute_map)
        PrintDecisionTree(decision_tree, 'ID3')
        result_list.append(
            Classify(org_test_set, decision_tree, 'ID3', attribute_set))
        print('ID3正确率：', AC(true_result, Classify(
            org_test_set, decision_tree, 'ID3', attribute_set)))

        train_set = [i for i in org_train_set]
        new_attribute_set = [i for i in attribute_set]

        decision_tree = DecisionTree(
            train_set, new_attribute_set, 'C4.5', attribute_map)
        PrintDecisionTree(decision_tree, 'C4.5')
        result_list.append(
            Classify(org_test_set, decision_tree, 'C4.5', attribute_set))
        print('C4.5正确率：', AC(true_result, Classify(
            org_test_set, decision_tree, 'C4.5', attribute_set)))

        train_set = [i for i in org_train_set]
        new_attribute_set = [i for i in attribute_set]

        decision_tree = DecisionTree(
            train_set, new_attribute_set, 'CART', attribute_map)
        PrintDecisionTree(decision_tree, 'CART')
        result_list.append(
            Classify(org_test_set, decision_tree, 'CART', attribute_set))
        print('CART正确率：', AC(true_result, Classify(
            org_test_set, decision_tree, 'CART', attribute_set)))
        n -= 1
    true_result = [i[-1] for i in org_test_set]
    count = 0
    for i in range(len(true_result)):
        answer = [line[i] for line in result_list]
        if max(answer, key=answer.count) == true_result[i]:
            count += 1

    print("最终随机森林得到的结果正确率：", count/len(true_result))

    # print("正确率如下")
    #print("占比      ID3       C4.5      CART")
    # for item in AC_list:
    # print("%-10f%-10f%-10f%-10f"
    # #%(item[0],item[1],item[2],item[3]))
