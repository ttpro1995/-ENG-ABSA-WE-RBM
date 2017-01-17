# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from labeling import Write2file
from os.path import splitext
import CONSTANT

def make_raw_sentiment_file (data,pos_neg_labels, folder_dir):
    file_raw_data = open(folder_dir+ '/'+CONSTANT.full_sentiment_data_raw,'w')
    file_raw_labels = open(folder_dir+'/'+ CONSTANT.full_sentiment_labels_raw,'w')
    for i in range(len(data)):
        sentence = data[i].rstrip() # remove all \n on right
        sentence = sentence.lstrip() # remove all \n on left
        file_raw_data.write(sentence + '\n')
        file_raw_labels.write(str(pos_neg_labels[i])  + '\n')


def strip_between(s, start = '>', end = '</'):
    """
    Return the word between start and end
    :param start:
    :param end:
    :return:
    """
    return s[s.index(start)+1:s.index(end)]

def strip_tag(s):
    number = strip_between(s, '<','>') # get the number in tag
    start = number+'>' # 2>
    return s[s.index(start)+2:(len(s)-len(start))]


def parse_raw_corpus_to_xml(filename, no_neutral = False):
    """
    Read Classified_Corpus.xml (all in lowercase) and parse to suittable xml
    :param filename:
    :return:
    """
    # const
    _FOOD = 'F'
    _STAFF = 'S'
    _AMBIENCE = 'A'
    _POS = '1'
    _NEG = '2'
    _NEU = '3'
    ###

    # store all data here, then write to file later
    index = 0
    all_labels =[]
    all_review_lines = []

    soup = BeautifulSoup(open(filename), "lxml")
    all_review = soup.find_all('review')
    for rv in all_review:
        for line in rv:
            # each review have multiple sentence
            POSITIVE = 0
            NEGATIVE = 0
            NEUTRAL = 0

            FOOD = 0
            STAFF = 0
            AMBIENCE = 0

            if ('<food>' in str(line)):
                FOOD = 1

            if ('<staff>' in str(line)):
                STAFF = 1
            if ('<Ambience>' in str(line)):
                AMBIENCE = 1

            if ('<positive>' in str(line)):
                POSITIVE = 1
            if ('<negative>' in str(line)):
                NEGATIVE = 1
            if ('<neutral>' in str(line)):
                NEUTRAL = 1

            if (POSITIVE+NEGATIVE+NEUTRAL == 0):
                continue

            if (FOOD + STAFF + AMBIENCE != 1):
                "we only use sentences with a single label for evaluation to avoid ambiguity"
                continue

            if (no_neutral):
                if (NEUTRAL):
                    "We will not use neutral"
                    continue
                if (POSITIVE+NEGATIVE == 2):
                    "It is neutral, we do not use"
                    continue
            elif (not no_neutral and (POSITIVE+NEGATIVE)==2):
                NEUTRAL = 1

            label_aspect = None
            label_sent = None

            if (FOOD):
                label_aspect = _FOOD
            elif(STAFF):
                label_aspect = _STAFF
            elif(AMBIENCE):
                label_aspect = _AMBIENCE

            if  (POSITIVE):
                label_sent = _POS
            elif (NEGATIVE):
                label_sent = _NEG
            elif (NEUTRAL):
                label_sent = _NEU


            text_line = line.get_text()
            text_line = strip_tag(text_line)
            all_labels.append(label_aspect+label_sent)
            all_review_lines.append(text_line)



    print ('start writing to file')

    Write2file(all_labels,all_review_lines,index, CONSTANT.DATASET_FOLDER_DIR)

def convert_to_lower_case(filename):
    """
    Convert everything in a file to lowercase
    :param filename:
    :return:
    """
    filename, file_extension = splitext(filename)
    file_output = open(filename+"_lower"+file_extension, 'w')
    with open(filename+file_extension, 'r') as fileinput:
        for line in fileinput:
            line = line.lower()
            file_output.write(line)


def load_data_sentiment (filename):
    """
    This use same format with Vietnamese dataset

    :param filename:
    :return:
    """
    file = open(filename,'r')
    data = file.read()
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

if __name__ == "__main__":
    # convert Classified Corpus to lower case first
    # convert_to_lower_case('../dataset/Classified_Corpus.xml')

    # parse corpus into xml
    # parse_raw_corpus_to_xml('../dataset/Classified_Corpus_lowercase.xml')


    # run on real corpus
    convert_to_lower_case(CONSTANT.DATASET_FOLDER_DIR+'/'+CONSTANT.Classified_Corpus)
    parse_raw_corpus_to_xml(CONSTANT.DATASET_FOLDER_DIR+'/'+CONSTANT.Classified_Corpus_lower)
    data, labels, posnegs = load_data_sentiment('../dataset/Output_FSA.txt')
    make_raw_sentiment_file(data,posnegs, CONSTANT.DATASET_FOLDER_DIR)

    # run on short corpus
    #convert_to_lower_case('../dataset/short_corpus')
    #parse_raw_corpus_to_xml('../dataset/short_corpus_lower')
    #data, labels, posnegs = load_data_sentiment('../dataset/Output_FSA.txt')
    #make_raw_sentiment_file(data,posnegs, DATASET_FOLDER_DIR)
