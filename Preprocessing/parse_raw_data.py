# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from labeling import Write2file

def make_raw_sentiment_file (data,pos_neg_labels):
    file_raw_data = open('full_sentiment_data_raw.txt','w')
    file_raw_labels = open('full_sentiment_labels_raw.txt','w')
    for i in range(len(data)):
        file_raw_data.write(data[i] + '\n')
        file_raw_labels.write(str(pos_neg_labels[i])  + '\n')


def parse_raw_corpus_to_xml(filename):
    """
    Read Classified_Corpus.xml (all in lowercase) and parse to suittable xml
    :param filename:
    :return:
    """
    soup = BeautifulSoup(open(filename), "lxml")
    all_review = soup.find_all('review')
    rv=all_review[0]
    for line in rv:
        # each review have multiple sentence
        POSITIVE = 0
        NEGATIVE = 0
        NEUTRAL = 0

        FOOD = 0
        STAFF = 0
        AMBIENCE = 0

        if (FOOD+STAFF+AMBIENCE != 1):
            "we only use sentences with a single label for evaluation to avoid ambiguity"
            continue



        print ('breakpoint')
    print ('breakpoint')

def convert_to_lower_case(filename):
    """
    Convert everything in a file to lowercase
    :param filename:
    :return:
    """
    file_output = open(filename+"lower", 'w')
    with open(filename, 'r') as fileinput:
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
    # convert_to_lower_case('./dataset/short_corpus')
    # parse_raw_corpus_to_xml('./dataset/short_corpuslower')
    Write2file(['F1','A2','F2'],["meow f1", "woof a2","meow f2"],113)
    #data, labels, posnegs = load_data_sentiment('./dataset/Classified_Corpus.xml')
    #make_raw_sentiment_file(data,posnegs)
