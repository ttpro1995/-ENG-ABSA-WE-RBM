#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator
from bs4 import BeautifulSoup
import gensim
from nltk.corpus import stopwords
try:
    import PIL.Image as Image
except ImportError:
    import Image

# Chuẩn hóa dữ liệu
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

    # Trả về kiểu list, mỗi phần tử trong list là một chuỗi, ví dụ ['boy works best company world', 'abc xyz']
    return new_training_data

# Kiểu classifier này sẽ giúp gợi ý cho RBM câu nào có tổng distance từ NGUYÊN 1 câu đến từ aspect nào là nhỏ nhất
# Dành cho aspect
def Word2Vec_classifier_1 (w2v_model, sentences, vectors, aspect_size,number_in_training_set):
    return_vectors = []
    for i in range(len(sentences[number_in_training_set:])):
        i = i + number_in_training_set
        words = sentences[i].split()
        words_in_vocabs = []
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                words_in_vocabs.append(words[j])
        if (words_in_vocabs != []):
            food_point = w2v_model.n_similarity(words_in_vocabs, ['food','drink'])
            staff_point = w2v_model.n_similarity(words_in_vocabs, ['staff','service'])
            ambience_point = w2v_model.n_similarity(words_in_vocabs, ['ambience','environment'])
            if (food_point >= staff_point and food_point >= ambience_point):
                return_vectors.append(1)
            if (staff_point >= food_point  and staff_point >= ambience_point):
                return_vectors.append(5)
            if (ambience_point >= staff_point and ambience_point >= food_point):
                return_vectors.append(3)
        else:
            return_vectors.append(-1)
    # Ghi ra file
    file = open('word2vec_predict_suggestion.txt','w')
    for i in range(len(return_vectors)):
        file.write(str(return_vectors[i]))
        file.write('\n')

# Dành cho sentiment
def Word2Vec_classifier_2 (w2v_model, sentences, vectors, aspect_size,number_in_training_set):
    return_vectors = []
    for i in range(len(sentences[number_in_training_set:])):
        i = i + number_in_training_set
        words = sentences[i].split()
        words_in_vocabs = []
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                words_in_vocabs.append(words[j])
        if (words_in_vocabs != []):
            positive_point = w2v_model.n_similarity(words_in_vocabs, ['perfect','lovely','worthy','like','good','delicious','fast','positive','kind','happy','fantastic','accept','beautiful','joyful','cheap'])
            negative_point = w2v_model.n_similarity(words_in_vocabs, ['dislike','bad','awful','slow','negative','expensive','unhappy','poor','terrible','boring','upset','sad','nasty','faulty','horrible'])
            if (positive_point >= negative_point):
                return_vectors.append(0)
            if (negative_point > positive_point):
                return_vectors.append(1)
        else:
            return_vectors.append(-1)
    # Ghi ra file
    file = open('word2vec_sentiment_predict_suggestion.txt','w')
    for i in range(len(return_vectors)):
        file.write(str(return_vectors[i]))
        file.write('\n')

# if __name__ == '__main__':
#     print "Loading Word2Vec pretrained model..."
#     w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format('GoogleNews.bin', binary=True)  # C binary format
#
#     Word2Vec_classifier_1 (w2v_model,
#         ['Their pad penang is delicious and everything else is fantastic','The dark red tone of the walls make the ambience kind of depressing','An authentic French , with a charming atmosphere ','He was highly inattentive and clearly ignored other table attempts as well as our own to catch his attention '])