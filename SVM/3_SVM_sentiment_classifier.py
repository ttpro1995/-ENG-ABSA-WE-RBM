#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from svmutil import *
from sklearn.model_selection import train_test_split
import CONSTANT
# Hàm load dữ liệu

def LoadData (filename1, filename2):
    file_data = open(filename1,'r')
    file_labels = open(filename2,'r')
    data = file_data.read().splitlines()
    labels = file_labels.read().splitlines()
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return data, labels

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


# Hàm chuẩn hóa nhãn từ string thành int. Check cụm từ trong labels, nếu có cụm từ đó thì tạo nhãn
def NormailizeLabels(training_labels):
    # Mảng chứa nhãn mới
    new_labels = []
    # Xét từng đoạn review
    for i in range(len(training_labels)):
        # Trong mỗi đoạn xét từng câu review
        for j in range(len(training_labels[i])):
        # Nhãn mới
            label = ''
            if ('RESTAURANT' in training_labels[i][j]):
                label = label + ' 0'
            if ('FOOD' in training_labels[i][j]):
                label = label + ' 1'
            if ('DRINKS' in training_labels[i][j]):
                label = label + ' 2'
            if ('AMBIENCE' in training_labels[i][j]):
                label = label + ' 3'
            if ('PRICES' in training_labels[i][j]):
                label = label + ' 4'
            if ('SERVICE' in training_labels[i][j]):
                label = label + ' 5'
            if ('Others' in training_labels[i][j] or label == ''):
                label = label + ' 6'
            new_labels.append(label)
    return new_labels

# Hàm chuân hóa dữ liệu: loại stopword, lower case nó xuống

def NormalizeData (training_data):
    # Định nghĩa biến trả về
    new_training_data = training_data
    # Đọc file stopword có sẵn trong thư viện lưu vào cache
    cachedStopWords = stopwords.words("english")

    for i in range(len(training_data)):
        # Viết thường
        temp = training_data[i].lower()
        # Loại stopword
        temp = ' '.join([word for word in temp.split() if word not in cachedStopWords])
        # Gán vào biến trả về
        new_training_data[i]= temp

    return new_training_data

def Readfile(file):
    data = []
    lines = file.read().splitlines()

    for i in range(len(lines)):
        values = lines[i].split('\t')
        sentence_class = int(values[0])
        sentence_value = values[1]
        temp = [sentence_value ,sentence_class ]
        data.append(temp)
    return data

def Convert_word2Idx (raw_data):
    # Building a dictionary
    Dict = []
    for i in range(len(raw_data)):
        # Split word in each line
        words = raw_data[i][0].split(' ')
        Dict = Dict + words
        # Eliminate duplicate elements
        Dict = list(set(Dict))
    # Now we have dictionary, each word in dictionary have its index, is also the index of Dictionary
    # Here we convert training sentence into <sentence class 1 -1> <word index>:<word class - 0 1>
    # Class of each sentence
    y = []
    for i in range(len(raw_data)):
        y.append(raw_data[i][1])
    # Sentences in right format
    x = []
    for i in range(len(raw_data)):
        # Init a small dict to contain words in one sentence that appear or not
        sent_dict = {}

        words_sent = raw_data[i][0].split(' ')
        for j in range(len(words_sent)):
            if (words_sent[j] in Dict):
                index = Dict.index(words_sent[j])
                sent_dict[index] = 1
        # Sort the dict and append it to data
        sort_sent_dict = collections.OrderedDict(sorted(sent_dict.items()))
        x.append(sort_sent_dict)

def LoadWord2VecLabels(filename):
    file = open(filename,'r')
    data= file.read().splitlines()
    for i in range(len(data)):
        data[i] = int(data[i])
    return data

def LoadWord2VecLabels_str(filename):
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

# Convert POS-tagged data
def Convert_word2Idx_NoPOStagged (nopostagged_data):
    # Building a dictionary
    Dict = []
    # Building a data in the appreciate form for LIBSVM
    Data = []
    for i in range(len(nopostagged_data)):
        # Split word in each line
        word = nopostagged_data[i].split(' ')
        Dict = Dict + word
        Data.append(word)
    # Eliminate duplicate elements
    Dict = list(set(Dict))
    print "    - Complete Dictionary"
    #print Data

    #Save dictionary
    file_dic = open("Dictionary.txt",'w')
    for i in range(len(Dict)):
        file_dic.write(str(Dict[i]) + "\n")
    # Sentences in right format
    x = []
    for i in range(len(Data)):
        sent_dict = {}
        # Check word
        for j in range(len(Data[i])):
            # Check if a word appeared in dictionary
            if (Data[i][j] in Dict):
                index = Dict.index(Data[i][j])
                sent_dict[index] = 1

        # Sort the dict and append it to data
        sort_sent_dict = collections.OrderedDict(sorted(sent_dict.items()))
        x.append(sort_sent_dict)
    return x

def Convert_word2Idx_NoPOStagged_predict (nopostagged_data):
    # Reading dictionary
    file_dict = open("Dictionary.txt",'r')
    Dict = file_dict.read().splitlines()

    Data = []
    for i in range(len(nopostagged_data)):
        # Split word in each line
        word = nopostagged_data[i].split(' ')
        Data.append(word)

    # Sentences in right format
    x = []

    for i in range(len(Data)):
        sent_dict = {}
        # Check word
        for j in range(len(Data[i])):
            # Check if a word appeared in dictionary
            if (Data[i][j] in Dict):
                index = Dict.index(Data[i][j])
                sent_dict[index] = 1

        # Sort the dict and append it to data
        sort_sent_dict = collections.OrderedDict(sorted(sent_dict.items()))
        x.append(sort_sent_dict)
    return x

def Make_raw_sentiment_file (data,pos_neg_labels):
    file_raw_data = open('full_sentiment_data_raw.txt','w')
    file_raw_labels = open('full_sentiment_labels_raw.txt','w')
    for i in range(len(data)):
        file_raw_data.write(data[i] + '\n')
        file_raw_labels.write(str(pos_neg_labels[i])  + '\n')

if __name__ == "__main__":

    data, labels = LoadData(CONSTANT.DATASET_FOLDER_DIR+'/'+ CONSTANT.full_sentiment_data_raw,
                            CONSTANT.DATASET_FOLDER_DIR+'/'+ CONSTANT.full_sentiment_labels_raw)
    # data, labels = LoadData('svm_data.txt', 'full_sentiment_labels.txt')

    data_size = 1500
    data = data[:data_size]
    labels = labels[:data_size]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    training_set = X_train
    train_labels = y_train


    # Make dictionary
    print "Making Dictionary..."
    training_set = Convert_word2Idx_NoPOStagged (X_train)
    # Tạo model
    print "Making a model"
    prob  = svm_problem(train_labels, training_set)
    param = svm_parameter('-t 0 -c 1 -b 1')
    # Train the model
    print "Training..."
    model = svm_train(prob, param)
    # Save the model
    svm_save_model('SVM_sentiment.model', model)



    # Kiểm thử, xuất file predict label, cho biết accuracy
    model = svm_load_model('SVM_sentiment.model')
    test_set = Convert_word2Idx_NoPOStagged_predict(X_test)
    test_labels = y_test
    # Predict
    p_label, p_acc, p_val = svm_predict(test_labels, test_set, model, '-b 1')
    ACC, MSE, SCC = evaluations(test_labels, p_label)

    file = open('result_SVM_sentiment.txt','w')
    file.write("Accuracy = " + str(ACC))