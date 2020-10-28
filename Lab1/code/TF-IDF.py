import math

#读取数据，根据空格将文档处理成词的集合
def loadData():
    file = open('./lab1_data/semeval.txt', 'r')
    sentence = []
    while True:
        text = file.readline()
        begin = text.rfind(':')
        text = text[begin:]
        for char in text:
            if char.isalpha() == True:
                begin = text.index(char)
                break
        text = text[begin:]
        if len(text.split()) > 0:
            sentence.append(text.split())
        if not text:
            break
    file.close()
    return sentence

#计算tf-idf矩阵
def tf_idf(sentence):
    count_dict = dict()
    word_set = set()
    word_tf = []
    #建立词库再按字典序进行排序
    for words in sentence:
        for word in words:
            word_set.add(word)
    word_set = sorted(word_set)
    for words in sentence:
        word_count = []
        for word in word_set:
            #防止文档为空，出现除0情况
            if len(words) == 0:
                word_count.append(words.count(word))
            #归一化的频率表示
            else:
                word_count.append((words.count(word))/len(words))
            #使用字典统计每个词出现的文档数
            if words.count(word) > 0:
                if word in count_dict:
                    count_dict[word] +=1
                else:
                    count_dict[word] = 1
        word_tf.append(word_count)
    #计算逆向文档频率
    for word in count_dict:
        count_dict[word] = math.log(len(sentence)/(count_dict[word]+1))
    #计算tf-idf矩阵
    for document in word_tf:
        for index in range(len(document)):
            document[index] = document[index] * count_dict[word_set[index]]
    return word_tf

if __name__ == '__main__':
    sentence = loadData()
    word_tf = tf_idf(sentence)
    data = open("18340236_ZhuYu_TFIDF.txt","w")
    print(word_tf,file=data)
    data.close()
