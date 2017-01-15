#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from svmutil import *
# Hàm load dữ liệu
def LoadData (filename1, filename2):
    file_data = open(filename1,'r')
    file_labels = open(filename2,'r')
    data = file_data.read().splitlines()
    labels = file_labels.read().splitlines()
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return data, labels

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
        Dict = list(set(Dict))

    y = []
    for i in range(len(raw_data)):
        y.append(raw_data[i][1])
  
    x = []
    for i in range(len(raw_data)):
     
        sent_dict = {}

        words_sent = raw_data[i][0].split(' ')
        for j in range(len(words_sent)):
            if (words_sent[j] in Dict):
                index = Dict.index(words_sent[j])
                sent_dict[index] = 1

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
        for line in f:  
            numbers_str = line.split()
            #convert numbers to floats
            numbers_float = [float(x) for x in numbers_str]  
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
    print "Number of Vocabulary: "
    print len(Dict)
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

if __name__ == "__main__":
    # Define
    number_in_training_set = 60582

    print "Loading Data..."
    # Load dữ liệu
    data,labels = LoadData('full_data_raw.txt','full_labels_raw.txt')

    training_set =  data[:60000]
    training_labels =  labels[:40] + labels[38694:38734] + labels[52876:52916]

    # Make dictionary
    print "Making Dictionary..."
    training_set = Convert_word2Idx_NoPOStagged (training_set)
  
    print "================================="

