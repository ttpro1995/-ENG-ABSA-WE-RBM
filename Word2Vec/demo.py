#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Dai Nguyen'
# ==========================================
# Script: Thực hiện thí nghiệm trên tiếng Việt

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
    data[0] = data[0].lower().replace(",", "").replace(".", "").split()

    return data[0]

# Hàm so sánh với food, staff, ambience
def Compare(w2v_model,data):
    new_data = []
    for i in range(len(data)):
        if data[i] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
            new_data.append(data[i])
    data = new_data
    food_point= 0
    staff_point= 0
    ambience_point = 0

    if( data != [] ):

        food_point = w2v_model.n_similarity(['thức', 'ăn','món','nước'],data)
        staff_point = w2v_model.n_similarity(['nhân', 'viên', 'phục','vụ'], data)
        ambience_point = w2v_model.n_similarity(['phong', 'cách', 'trang','trí','quán','nhà','hàng'],data)

    print "Khía cạnh: "

    if(food_point>0.2):
        temp = []
        for i in range(len(data)):
            if (w2v_model.n_similarity(['thức', 'ăn','món','nước','ngon','đẹp','tuyệt','hấp','dẫn','dở','tệ','nhạt','cứng'], [data[i]]) > 0.25):
                temp.append(data[i])
                if ((i+1) in range(len(data))):
                    temp.append(data[i+1])
        food_point_positive = 0
        food_point_negative = 0
        if (temp != []):
            food_point_positive = w2v_model.n_similarity(['ngon','đẹp','tuyệt','hấp','dẫn'], temp)
            food_point_negative = w2v_model.n_similarity(['dở','tệ','nhạt','cứng'], temp)
        if (food_point_positive >= food_point_negative and 'không' not in data):
            print "- Thức ăn --> tích cực\n"
        elif (food_point_positive < food_point_negative and 'không' not in data):
            print "- Thức ăn --> tiêu cực\n"

        if (food_point_positive >= food_point_negative and 'không' in data):
            print "- Thức ăn --> tiêu cực\n"
        elif (food_point_positive < food_point_negative and 'không' in data):
            print "- Thức ăn --> tích cực\n"

    if(staff_point>0.4):
        temp = data
        temp = []
        for i in range(len(data)):
            if (w2v_model.n_similarity(['nhân', 'viên', 'phục','vụ','nhanh','thân','tận','tình','tốt','chậm','lạnh','thái','độ','phiền','tệ'], [data[i]]) > 0.25):
                temp.append(data[i])
                if ((i+1) in range(len(data))):
                    temp.append(data[i+1])
        staff_point_positive = 0
        staff_point_negative = 0
        if (temp != []):
            staff_point_positive = w2v_model.n_similarity(['nhanh','thân','tận','tình','tốt'], temp)
            staff_point_negative = w2v_model.n_similarity(['chậm','lạnh','thái','độ','phiền','tệ'], temp)
        if (staff_point_positive >= staff_point_negative and 'không' not in data):
            print "- Nhân viên --> tích cực\n"
        elif (staff_point_positive < staff_point_negative and 'không' not in data):
            print "- Nhân viên --> tiêu cực\n"

        if (staff_point_positive >= staff_point_negative and 'không' in data):
            print "- Nhân viên --> tiêu cực\n"
        elif (staff_point_positive < staff_point_negative and 'không' in data):
            print "- Nhân viên --> tích cực\n"


    if(ambience_point>0.4):
        temp = data
        for i in range(len(data)):
            if (w2v_model.n_similarity(['phong', 'cách', 'trang','trí','quán','nhà','hàng','đẹp','ấm','cúng','yên','tĩnh','dễ','xấu','dơ','bừa','bộn','đông','khó'], [data[i]]) > 0.25):
                temp.append(data[i])
                if ((i+1) in range(len(data))):
                    temp.append(data[i+1])
        ambience_point_positive = 0
        ambience_point_negative = 0
        if (temp != []):
            ambience_point_positive = w2v_model.n_similarity(['đẹp','ấm','cúng','yên','tĩnh','dễ'],temp)
            ambience_point_negative = w2v_model.n_similarity(['xấu','dơ','bừa','bộn','đông','khó'],temp)
        if (ambience_point_positive >= ambience_point_negative and 'không' not in data):
            print "- Trang trí --> tích cực\n"
        elif (ambience_point_positive < ambience_point_negative and 'không' not in data):
            print "- Trang trí --> tiêu cực\n"

        if (ambience_point_positive >= ambience_point_negative and 'không' in data):
            print "- Trang trí --> tiêu cực\n"
        elif (ambience_point_positive < ambience_point_negative and 'không' in data):
            print "- Trang trí --> tích cực\n"

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
    while (1):
        data=[]
        data.append(raw_input("Nhập câu đánh giá: "))
        if (data[0] == "0"):
            break
        # Load mô hình
        w2v_model = gensim.models.word2vec.Word2Vec.load('Word2Vec_TiengViet.model')
        data_vector = Word2Vec(data, w2v_model)

        # So sánh với vector food, staff, ambience
        predict_labels = Compare(w2v_model,data_vector)

