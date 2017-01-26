#!/usr/bin/python
# -*- coding: utf-8 -*-
# Script chạy trên py32 (không sử dụng anaconda)

from __future__ import print_function
import collections
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from svmutil import *
from sklearn.model_selection import train_test_split
import time
import CONSTANT
import util.log_util



# Hàm load dữ liệu
def LoadData (filename1, filename2):
    file_data = open(filename1,'r')
    file_labels = open(filename2,'r')
    data = file_data.read().splitlines()
    labels = file_labels.read().splitlines()
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return data, labels

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
    print ("    - Complete Dictionary")
    #print Data
    print ("Number of Vocabulary: ")
    print (len(Dict))
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

    # create logger
    logger = util.log_util.create_logger("SVM_aspect_classifier", print_console=True)
    print = logger.info # redirect print function to logger.info
    print ('Language %s' %(CONSTANT.LANGUAGE))

    # Define
    data_size = 0 # set data_size = 0 to train on all data


    print ("Loading Data...")
    # Load dữ liệu
    data, labels = LoadData(CONSTANT.DATASET_FOLDER_DIR+'/'+ CONSTANT.full_aspect_data_raw,
                            CONSTANT.DATASET_FOLDER_DIR+'/'+ CONSTANT.full_aspect_labels_raw)
    print (len(labels))
    if (data_size>0):
        data = data[:data_size]
        labels = labels[:data_size]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
    print("Data: %d train sample, %d test sameple" %(len(y_train), len(y_test)))

    # training_set =  data[:40] + data[38694:38734] + data[52876:52916]
    training_set =  X_train
    training_labels =  y_train

    # Make dictionary
    print ("Making Dictionary...")
    training_set = Convert_word2Idx_NoPOStagged (training_set)

    # Tạo model
    print ("Making a model")
    prob  = svm_problem(training_labels, training_set)
    param = svm_parameter('-t 0 -c 1 -b 1')
    # Train the model
    print ("Training...")
    start_training_time = time.time()
    model = svm_train(prob, param)
    # Save the model
    svm_save_model('SVM_aspect.model', model)



    # Kiểm thử, xuất file predict label, cho biết accuracy
    model = svm_load_model('SVM_aspect.model')
    test_set = Convert_word2Idx_NoPOStagged_predict(X_test)
    test_labels = y_test
    done_train_time = time.time()
    print ('Training in %f' %(done_train_time-start_training_time))
    # Predict
    print ("Predict ...")
    p_label, p_acc, p_val = svm_predict(test_labels, test_set, model, '-b 1')
    ACC, MSE, SCC = evaluations(test_labels, p_label)

    # Ghi file predict labels
    file = open('svm_aspect_predict.txt','w')
    for i in range(len(p_label)):
        file.write(str(p_label[i]) + '\n')

    # Tìm Precision, Recall và F1
    true_labels = test_labels
    predict_labels = p_label

    print (predict_labels[0:10])
    print (true_labels[0:10])
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
    print ("For Food Aspect =================")
    print ('Precision is: ')
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print (pre)

    print ('Recall is: ')
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print (recall)

    print ('F1 is: ')
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print (f1)
    print ("=================================")

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
    print ("For Staff Aspect =================")
    print ('Precision is: ')
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print (pre)

    print ('Recall is: ')
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print (recall)

    print ('F1 is: ')
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print (f1)
    print ("=================================")

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
    print ("For Ambience Aspect =================")
    print ('Precision is: ')
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)*1.0
    print (pre)

    print ('Recall is: ')
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)*1.0
    print (recall)

    print ('F1 is: ')
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)*1.0
    print (f1)
    print ("=================================")

