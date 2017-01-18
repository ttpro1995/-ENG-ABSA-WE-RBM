#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Thai Thien'
# ==========================================
# Testing on Word2Vec

import numpy as np
import operator
from bs4 import BeautifulSoup
import gensim
from scipy import spatial
from nltk.corpus import stopwords
try:
    import PIL.Image as Image
except ImportError:
    import Image

# Hàm load dữ liệu
def LoadData (filename):
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
    # Có nhãn rồi chuyển nó sang labels (dạng số) được quy định như trên
    for i in range(len(all_sentences)):
        if ('FOOD' in all_aspects[i]):
            all_labels[i] += ' 1'
        if ('AMBIENCE' in all_aspects[i]):
            all_labels[i] += ' 3'
        if ('STAFF' in all_aspects[i]):
            all_labels[i] += ' 5'

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
        if (all_labels[i] == ' 1'):
            text = all_sentences[i].text
            label = 1
            data.append(text)
            labels.append(label)
            posnegs.append(all_posneg_labels[i])
        if (all_labels[i] == ' 5'):
            text = all_sentences[i].text
            label = 5
            data.append(text)
            labels.append(label)
            posnegs.append(all_posneg_labels[i])
        if (all_labels[i] == ' 3'):
            text = all_sentences[i].text
            label = 3
            data.append(text)
            labels.append(label)
            posnegs.append(all_posneg_labels[i])
    return data, labels, posnegs

# Hàm chuyển câu thành vector bằng công cụ word2vec
def Word2Vec(data,w2v_model):

    # Chuyển câu thành vector.........
    for i in range(len(data)):
        # Khai báo vector cho 1 câu, số chiều mặc định là 300
        vector = np.zeros(300)
        words = data[i].split()
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                vector = vector + w2v_model[words[j]] # Cộng hết tất cả vector của từ lại sẽ tạo thành vector của câu
        if (i==0):
            data_vector = vector
        else:
            data_vector = np.vstack((data_vector,vector))

    return data_vector

# Hàm so sánh với food, staff, ambience
def Compare(w2v_model,data_vector, num_in_training_set):
    result_vector = []
    food_vector = w2v_model['thức']+w2v_model['ăn']
    staff_vector = w2v_model['nhân']+w2v_model['viên']
    ambience_vector = w2v_model['không']+w2v_model['gian']

    for i in range(len(data_vector[num_in_training_set:])):
        i = i + num_in_training_set
        foodpoint = 1 - spatial.distance.cosine(data_vector[i][:300], food_vector)
        staffpoint = 1 - spatial.distance.cosine(data_vector[i][:300], staff_vector)
        ambiencepoint = 1 - spatial.distance.cosine(data_vector[i][:300], ambience_vector)

        if (foodpoint > staffpoint and foodpoint > ambiencepoint):
            result_vector.append(1)
        elif (staffpoint > foodpoint and staffpoint > ambiencepoint):
            result_vector.append(5)
        elif (ambiencepoint > staffpoint and ambiencepoint > foodpoint):
            result_vector.append(3)
        else:
            result_vector.append(0)
    return result_vector

def LoadWord2VecLabels(filename):
    file = open(filename,'r')
    data= file.read().splitlines()
    return data
def LoadWord2VecData(filename):
    data = []
    with open(filename) as f:
        for line in f:  #Line is a string
            #split the string on whitespace, return a list of numbers
            # (as strings)
            numbers_str = line.split()
            #convert numbers to floats
            numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
            data.append(numbers_float)
    return data

if __name__ == '__main__':
    # Load dữ liệu sentences
    print "Loading Data..."
    # Load dữ liệu
    data, labels, pos_neg_labels = LoadData('..\_Data\Output_FSA.xml')

    # Load mô hình
    w2v_model = gensim.models.word2vec.Word2Vec.load('Word2Vec_TiengViet.model')  # C binary format

    # Chuyển sang vector
    print "Word2Vec phrase..."
    data_vector = Word2Vec(data, w2v_model)

    # So sánh với vector food, staff, ambience
    predict_labels = Compare(w2v_model,data_vector,0)
    true_labels = labels

    # Kiểm tra
    # Food
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == 1 and true_labels[i] == 1):
            true_positive += 1
        elif (predict_labels[i] == 1 and true_labels[i] != 1) :
            false_positive += 1
        elif ((predict_labels[i] == 5 or predict_labels[i] == 3)  and true_labels[i] == 1) :
            false_negative += 1
    print "For Food Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print f1
    print "================================="

     # Staff
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == 5 and true_labels[i] == 5):
            true_positive += 1
        elif (predict_labels[i] == 5 and true_labels[i] != 5) :
            false_positive += 1
        elif ((predict_labels[i] == 1 or predict_labels[i] == 3)  and true_labels[i] == 5) :
            false_negative += 1
    print "For Staff Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print f1
    print "================================="

     # Ambience
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == 3 and true_labels[i] == 3):
            true_positive += 1
        elif (predict_labels[i] == 3 and true_labels[i] != 3) :
            false_positive += 1
        elif ((predict_labels[i] == 5 or predict_labels[i] == 1)  and true_labels[i] == 3) :
            false_negative += 1
    print "For Ambience Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print f1
    print "================================="
