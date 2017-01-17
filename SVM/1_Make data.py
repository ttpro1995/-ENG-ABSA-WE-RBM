#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Dai Nguyen'

# Script chạy với anaconda

#Thư viện
import numpy as np
import operator
from bs4 import BeautifulSoup
import gensim
from scipy import spatial
from nltk.corpus import stopwords
from Preprocessing.parse_raw_data import make_raw_aspect_file, make_raw_sentiment_file, load_data_sentiment_aspect
import CONSTANT

if __name__ == "__main__":
    data, aspect_labels, posnegs = load_data_sentiment_aspect('../dataset/Output_FSA.txt')
    make_raw_sentiment_file(data, posnegs, CONSTANT.DATASET_FOLDER_DIR)
    make_raw_aspect_file(data, aspect_labels, CONSTANT.DATASET_FOLDER_DIR)
