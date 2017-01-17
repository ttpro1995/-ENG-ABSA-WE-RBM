#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Hàm load dữ liệu
def LoadData_sentiment (filename):
    file = open(filename,'r')
    data= file.read()
    soup = BeautifulSoup(data, 'xml')
    all_sentences = soup.find_all('sentence')
    all_aspects = []
    all_labels = [''] * len(all_sentences)
    posneg_labels = []

    # Tìm aspect cho các câu đó
    for i in range(len(all_sentences)):
        if ('<Opinions>' in str(all_sentences[i])):
            Opinions = all_sentences[i].Opinions.find_all('Opinion')
            aspects = ''
            sentiments = ''
            for j in range(len(Opinions)):
                aspect = Opinions[j]['category']
                sentiment = Opinions[j]['polarity']
                aspects += ' ' + aspect
                sentiments += ' ' + sentiment
            all_aspects.append(aspects)
            posneg_labels.append(sentiments)
        else:
            all_aspects.append('Others')
            posneg_labels.append('Others')
    # Gán -1 cho tất cả vì ko cần nhãn aspect
    for i in range(len(all_sentences)):
        all_labels[i] += ' -1'

    all_posneg_labels = [0]*len(posneg_labels)
    for i in range(len(all_sentences)):
        if ('negative' in posneg_labels[i] and 'positive' in posneg_labels[i]):
            all_posneg_labels[i] = 2
        elif ('positive' in posneg_labels[i]):
            all_posneg_labels[i] = 0
        elif ('negative' in posneg_labels[i]):
            all_posneg_labels[i] = 1

    # Tìm ra những câu chỉ nói về food staff hoặc ambience
    data = []
    labels = []
    posnegs = []
    for i in range(len(all_sentences)):
        text = all_sentences[i].text
        data.append(text)
        labels.append(-1)
        posnegs.append(all_posneg_labels[i])

    return data, labels, posnegs

def ReadFileSentiWordNet(filename):
    senti_word = []
    senti_pos = []
    senti_neg = []
    file = open(filename,'r')
    full_data = file.read().splitlines()
    for i in range(len(full_data)): # Với mỗi dòng trong sentiwordnet
        columns = full_data[i].split('\t')

        words = columns[4].split(' ')
        # Xét mỗi từ
        for i in range(len(words)):
            # Bỏ 2 ký tự cuối
            words[i] = words[i][:-2]
            # Xét coi có trong senti_word chưa, nếu chưa có thêm vào
            if (words[i] not in senti_word):
                senti_word.append(words[i])
                senti_pos.append(float(columns[2]))
                senti_neg.append(float(columns[3]))
    return senti_word,senti_pos,senti_neg

def Senti_predict (data,  senti_words, senti_pos, senti_neg):
    predict_labels = []
    for i in range(len(data)):
        print "predict " + str(i) + "/" + str(len(data))
        pos_point = 0
        neg_point = 0
        words = data[i].split()
        for j in range(len(words)):
            if (words[j] in senti_words):
                word_index = senti_words.index(words[j])
                pos_point += senti_pos[word_index]
                neg_point += senti_neg[word_index]
        if (pos_point >= neg_point):
            predict_labels.append(0)
        else:
            predict_labels.append(1)
    return predict_labels

def Accuracy_SentiWordNet (predict_labels,pos_neg_labels):
    true_positive = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == 0 and pos_neg_labels[i] == 0):
            true_positive += 1
        if (predict_labels[i] == 1 and pos_neg_labels[i] == 1):
            true_positive += 1

    acc = true_positive*100.0/len(predict_labels)

    return acc
if __name__ == "__main__":
    # Load dữ liệu cần test lên (data + label)
    print "Load data and labels"
    data, labels, pos_neg_labels = LoadData_sentiment('full_data_sentiment_only.xml')

    # Chuẩn hóa dữ liệu


    # Load file SentiWordNet lên
    print "Loading SentiWordNet"
    senti_words, senti_pos, senti_neg = ReadFileSentiWordNet('SentiWordNet_3.0.0.txt')


    # Kiểm thử bằng SentiWordnet
    print "Predicting..."
    predict_labels = Senti_predict (data,  senti_words, senti_pos, senti_neg)

    # Tính accuracy và lưu file
    print "Accuracy is: "
    acc = Accuracy_SentiWordNet (predict_labels,pos_neg_labels)
    print acc

    file = open('result_sentiwordnet_sentiment.txt','w')
    file.write(str(acc))