#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Dai Nguyen'

#Thư viện
import gensim
import CONSTANT

def training_w2v_model():
    # Load file
    print "Loading file..."
    # file_training = open('Training_word2vec_viet.TXT','r')
    file_training = open(CONSTANT.DATASET_FOLDER_DIR+'/'+CONSTANT.full_sentiment_data_raw)
    sentences = file_training.readlines()

    # Huấn luyện mô hình Word2Vec cho tiếng Việt
    print "Splitting file..."
    for i in range(len(sentences)):
        sentences[i] = sentences[i].split()
    # print sentences
    print "Training model..."
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=300, window=5, min_count=1, workers=5)
    print "Save model..."
    model.save(CONSTANT.DATASET_FOLDER_DIR+'/'+ CONSTANT.Word2Vec_ENG_model)


if __name__ == "__main__":
    training_w2v_model()