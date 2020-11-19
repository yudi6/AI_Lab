from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data as Data
from xml.dom.minidom import parse
import gensim
import os
import csv
import gensim.downloader

torch.manual_seed(1)
use_gpu = torch.cuda.is_available()

# 使用Word2Vec训练词向量模型
def TrainWord2Vec(sentences, file, input_size, word_to_ix):
    weight = torch.zeros(len(word_to_ix), input_size)
    # 使用原有的Glove词向量
    if os.path.exists(file):
        Word2Vec_model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)
        Word2Vec_model_temp = gensim.models.Word2Vec(sentences, size=input_size, min_count=1)
        for i in range(len(Word2Vec_model_temp.wv.index2word)):
            word = Word2Vec_model_temp.wv.index2word[i]
            index = word_to_ix[word]
            if word in Word2Vec_model:
                weight[index, :] = torch.from_numpy(Word2Vec_model.wv.get_vector(word))
            else:
                weight[index, :] = torch.randn(input_size)
    # 使用Word2vec训练
    else:
        Word2Vec_model = gensim.models.Word2Vec(sentences, size=input_size, min_count=1)
        for i in range(len(Word2Vec_model.wv.index2word)):
            index = word_to_ix[Word2Vec_model.wv.index2word[i]]
            weight[index, :] = torch.from_numpy(Word2Vec_model.wv.get_vector(Word2Vec_model.wv.index2word[i]))
    # 模型根据word_to_ix 转换成Torch型矩阵
    return weight


def TruncAndPad(sentence_set, sentence_tag, max_len):
    # 句子长度大于设定长度则截断
    if (len(sentence_set) >= max_len):
        return sentence_set[0: max_len], sentence_tag[0:max_len]
    # 句子长度小于设定长度则在句子后增加“<PAD>”
    else:
        sentence_set.extend(["<PAD>"] * (max_len - len(sentence_set)))
        sentence_tag.extend(["<PAD>"] * (max_len - len(sentence_tag)))
    return sentence_set, sentence_tag


def GetSentenceTag(sentence_set, termset):
    # 返回的标签序列 初始化全为‘O’
    sentence_tag = ['O' for i in range(len(sentence_set))]
    for term in termset:
        # 对于每个关键词进行分词得到集合
        term_split = term.split()
        begin = term_split[0]
        # 关键短语的开始词标志为‘B’
        if begin in sentence_set:
            sentence_tag[sentence_set.index(begin)] = 'B'
        # 关键短语的其余词标志为‘I’
        for i in range(1, len(term_split)):
            in_t = term_split[i]
            if in_t in sentence_set:
                sentence_tag[sentence_set.index(in_t)] = 'I'
    return sentence_tag


def LoadData(file):
    train_data = []
    DOMTree = parse(file)
    sentence_list = DOMTree.documentElement
    # 读取每个样例
    sentences = sentence_list.getElementsByTagName('sentence')
    for sentence in sentences:
        # 样例句子文本
        sentence_text = sentence.getElementsByTagName('text')[0]
        sentence_term = sentence.getElementsByTagName('aspectTerm')
        # 对句子进行分词
        sentence_set = sentence_text.childNodes[0].data.replace('.', '').replace('-', '').lower().split()
        # 关键词集合
        term_set = []
        for Term in sentence_term:
            term_set.append(Term.getAttribute('term'))
        # 根据关键词集合得到标签序列
        sentence_tag = GetSentenceTag(sentence_set, term_set)
        # 对句子进行截断或增长
        train_data.append(TruncAndPad(sentence_set, sentence_tag, MAX_LEN))
    return train_data


# 返回每行的最大值
def ArgMax(Arg):
    num, idx = torch.max(Arg, 1)
    return idx.item()


# 根据to_ix将列表转化成Tensor
def Word2Tensor(word_set, to_ix):
    return torch.tensor([to_ix[i] for i in word_set], dtype=torch.long)


# batch版 根据to_ix将列表转化成Tensor
def Word2TensorBatch(train_set, word_to_ix, tag_to_ix):
    sentence_batch = torch.tensor([[word_to_ix[word] for word in line[0]] for line in train_set], dtype=torch.long)
    tags_batch = torch.tensor([[tag_to_ix[tag] for tag in line[1]] for line in train_set], dtype=torch.long)
    return sentence_batch, tags_batch


def LogSumExp(vector):
    max_score = vector[0, ArgMax(vector)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vector.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vector - max_score_broadcast)))


def LogSum(vector):
    return torch.log(torch.sum(torch.exp(vector), axis=0))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_rate, vocab_size, tag_to_ix, word_weight):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输出向量大小 也为词向量大小
            hidden_size=hidden_size // 2,  # 隐状态输出向量大小 双向则为1/2
            num_layers=num_layers,  # 层数
            bidirectional=True,  # 双向
            batch_first=True)  # 是否batch
        self.word_embeds = nn.Embedding(vocab_size, input_size)             # 采用随机初始化的词向量 并做训练
        # self.word_embeds = nn.Embedding.from_pretrained(word_weight)  # 采用训练的词向量
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden2tag = nn.Linear(hidden_size, self.tag_size)  # 线性层从隐状态向量到标签得分向量
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))  # CRF的转移矩阵表示从列序号对应标签转换到行序号对应标签
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 任意标签不能转移到start标签
        self.transitions.data[:, tag_to_ix[END_TAG]] = -10000  # end标签不能转移到任意标签
        self.hidden = self.HiddenInit()  # 隐藏层初始化

    def HiddenInit(self):
        return torch.randn(2, 1, self.hidden_size // 2), torch.randn((2, 1, self.hidden_size // 2))

    def ForwardAlg(self, feats):
        if use_gpu:
            init_alphas = torch.full([feats.shape[0], self.tag_size], -10000.).cuda()
        else:
            init_alphas = torch.full([feats.shape[0], self.tag_size], -10000.)
        # 开始标签的转换得分为0
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        # 输入的每个句子都进行前向算法
        forward_var_list = []
        forward_var_list.append(init_alphas)
        # 每个句子从句首开始迭代
        for feat_index in range(feats.shape[1]):
            # 迭代到某一词的logsumexp
            tag_score_now = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            feats_batch = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)
            # 新词的所有转移路径
            next_tag_score = tag_score_now + feats_batch + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(next_tag_score, dim=2))
        # 加上end标签的得分
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[END_TAG]].repeat([feats.shape[0], 1])
        # 每个句子进行logsumexp得到最终结果
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    # 单个句子经过lstm层加全连接层 输入为二位张量 变成三维张量
    def GetFeats(self, sentence):
        self.hidden = self.HiddenInit()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.squeeze()
        feats = self.hidden2tag(lstm_out)
        return feats

    # batch版 输入为三位张量
    def GetFeatsBatch(self, sentence):
        self.hidden = self.HiddenInit()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out)
        return feats

    # 给定序列tags的得分
    def Score(self, feats, tags):
        if use_gpu:
            score = torch.zeros(tags.shape[0]).cuda()
        else:
            score = torch.zeros(tags.shape[0])
        if use_gpu:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long().cuda(), tags], dim=1)
        else:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long(), tags], dim=1)
        for i in range(feats.shape[1]):
            # 第i个词得到的feats 二维张量
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[
                        range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[END_TAG], tags[:, -1]]
        return score

    # 维特比算法解码
    def Viterbi(self, feats):
        backpointers = []
        if use_gpu:
            init_vvars = torch.full((1, self.tag_size), -10000.).cuda()
        else:
            init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        # 三维的得分张量
        forward_var_list = []
        forward_var_list.append(init_vvars)
        # 从句子第一个词开始迭代
        for feat_index in range(feats.shape[0]):
            tag_score_now = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            tag_score_now = torch.squeeze(tag_score_now)
            next_tag_var = tag_score_now + self.transitions
            # 得到的最大得分
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)
            feat_batch = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + feat_batch
            forward_var_list.append(forward_var_new)
            # 标注得分的标签
            backpointers.append(bptrs_t.tolist())
        # 最终得分加入结束标志
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[END_TAG]]
        # 从后往前解码
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        # 根据tag_id进行解码
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        # 反转则是最终解码结果
        best_path.reverse()
        return path_score, best_path

    def LossFuction(self, sentences, tags):
        feats = self.GetFeatsBatch(sentences)
        forward_score = self.ForwardAlg(feats)
        gold_score = self.Score(feats, tags)
        # 所有输出句子的误差和作为结果
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):
        lstm_feats = self.GetFeats(sentence)
        score, tag_seq = self.Viterbi(lstm_feats)
        return score, tag_seq


if __name__ == '__main__':
    START_TAG = "<START>"
    END_TAG = "<END>"
    PAD_TAG = "<PAD>"
    INPUT_SIZE = 100  # 词向量大小
    HIDDEN_SIZE = 200  # 输出特征向量大小
    NUM_LAYERS = 2  # BiLSTM层数
    DROP_RATE = 0.5  # drop out rate
    EPOCH = 5000 # 迭代次数
    LR = 0.000001  # 学习率
    MAX_LEN = 30  # 句子长度
    train_data = LoadData(r'Laptops_Train.xml')
    trial_data = LoadData(r'laptops-trial.xml')
    word_to_ix = {"<PAD>": 0}
    all_sentence = []
    for sentence, tags in train_data:
        all_sentence.append(sentence)
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    for sentence, tags in trial_data:
        all_sentence.append(sentence)
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # 词向量文件
    WordFile = 'glove.6B.100d.txt'
    word_weight = TrainWord2Vec(all_sentence, WordFile, INPUT_SIZE, word_to_ix)
    tag_to_ix = {PAD_TAG: 0, "B": 1, "I": 2, "O": 3, START_TAG: 4, END_TAG: 5}
    # model = torch.load('model_vec2.pkl')
    model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROP_RATE, len(word_to_ix), tag_to_ix, word_weight)    # 模型
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)  # 优化器

    # loss_list = []
    # acc_list = []
    for epoch in tqdm(range(EPOCH)):
        model.zero_grad()
        sentence_in, targets = Word2TensorBatch(train_data, word_to_ix, tag_to_ix)
        if use_gpu:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()
        # 根据整个数据集进行训练
        loss = model.LossFuction(sentence_in, targets)
        loss.backward()
        # 梯度下降
        optimizer.step()
        # print(loss)
        # loss_list.append(loss.data.tolist())
        # 计算验证集准确率
    torch.save(model, 'model.pkl')
    with torch.no_grad():
        count = 0
        for line in trial_data:
            if use_gpu:
                sentence = Word2Tensor(line[0], word_to_ix).cuda()
            else:
                sentence = Word2Tensor(line[0], word_to_ix)
            if torch.equal(torch.tensor(model(sentence)[1]), Word2Tensor(line[1], tag_to_ix)):
                count += 1
        print(count / len(trial_data))
    # with open('save3.csv','w',newline='') as fp:
    #     write = csv.writer(fp)
    #     write.writerow(loss_list)
    #     write.writerow(acc_list)
    # model_temp = torch.load('model.pkl')
    # with torch.no_grad():
    #     count = 0
    #     for line in trial_data:
    #         sentence = Word2Tensor(line[0], word_to_ix).cuda()
    #         if torch.equal(torch.tensor(model_temp(sentence)[1]), Word2Tensor(line[1], tag_to_ix)):
    #             count += 1
    #     print(count / len(trial_data))
