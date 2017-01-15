#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from bs4 import BeautifulSoup
import gensim
from nltk.corpus import stopwords
from Word2Vec_classifier import Word2Vec_classifier_1
from Word2Vec_classifier import Word2Vec_classifier_2

try:
    import PIL.Image as Image
except ImportError:
    import Image

# Định nghĩa mô hình RBM

class RBM:

  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000):
 

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):
      # Clamp to the data and sample from the hidden units.
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_probs, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      print("Epoch %s: error is %s" % (epoch, error))
  def sampling(self,data):
    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Calculate the activations of the visible units.
    visible_activations = np.dot(hidden_probs, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_probs = visible_probs[:,1:]
    visible_states = visible_states[:,1:]
    return visible_probs

  def run_visible(self, data):
  
    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1

    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  def run_hidden(self, data):


    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states

  def daydream(self, num_samples):


    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

 
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

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

# Hàm load dữ liệu
def LoadData_aspect (filename):
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
        if ('RESTAURANT' in all_aspects[i]):
            all_labels[i] += ' 0'
        if ('FOOD' in all_aspects[i]):
            all_labels[i] += ' 1'
        if ('DRINKS' in all_aspects[i]):
            all_labels[i] += ' 2'
        if ('AMBIENCE' in all_aspects[i]):
            all_labels[i] += ' 3'
        if ('PRICES' in all_aspects[i]):
            all_labels[i] += ' 4'
        if ('SERVICE' in all_aspects[i]):
            all_labels[i] += ' 5'
        if ('Others' in all_aspects[i]):
            all_labels[i] += ' 6'

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

# Hàm chuyển file nhãn label thành dạng numpy để so sánh
def Convert2Np (labels,number_of_hidden_nodes):
    newlabels = []
    for i in range(len(labels)):
        np_label = [0]*number_of_hidden_nodes
        for j in range(number_of_hidden_nodes):
            if(str(j) in labels[i]):
                np_label[j] = 1
        newlabels.append(np_label)
    return np.array(newlabels)

# Hàm chuyển câu thành vector bằng công cụ word2vec
def Word2Vec_aspect(data,labels, pos_neg_labels,number_in_training_set,aspect_size, sentiment_size,w2v_model):

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

    # Thêm tất cả các câu (train lẫn test) những số 0 vào để đủ slot cho nhãn predict được
    for i in range(len(data)):
        # Tạo vector cần thêm vào
        add_vec = np.zeros(3*aspect_size)
        vector2 = np.append(data_vector[i],add_vec)
        if (i == 0):
            return_data_vector = vector2
        else:
            return_data_vector = np.vstack((return_data_vector,vector2))


    # Vị trí dành cho aspect
    for i in range(len(data[:number_in_training_set])):
        if (labels[i] == 1): # Food
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-2*aspect_size-1-j] = 1

        elif (labels[i] == 5): # Staff
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-aspect_size-1-j] = 1

        elif (labels[i] == 3): # Ambience
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-1-j] = 1
    return return_data_vector

def Word2Vec_sentiment(data,labels, pos_neg_labels,number_in_training_set,aspect_size, sentiment_size,w2v_model):

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

    # Thêm tất cả các câu (train lẫn test) những số 0 vào để đủ slot cho sentiment
    for i in range(len(data)):
        # Tạo vector cần thêm vào
        add_vec = np.zeros(2*sentiment_size)
        vector2 = np.append(data_vector[i],add_vec)
        if (i == 0):
            return_data_vector_sentiment = vector2
        else:
            return_data_vector_sentiment = np.vstack((return_data_vector_sentiment,vector2))


    # Vị trí dành cho sentiment
    for i in range(len(data[:number_in_training_set])):
        if (pos_neg_labels[i] == 0): # Positive
            for j in range(sentiment_size):
                return_data_vector_sentiment[i][return_data_vector_sentiment.shape[1]-sentiment_size-1-j] = 1

        elif (pos_neg_labels[i] == 1): # Negative
            for j in range(aspect_size):
                return_data_vector_sentiment[i][return_data_vector_sentiment.shape[1]-1-j] = 1

    return return_data_vector_sentiment

# Hàm main
def test_rbm(w2v_model,learningrate =0.1, training_epochs=20, number_in_training_set = 1000,
              aspect_size = 1, sentiment_size = 1,n_hidden=500):

 
    print "Loading Data..."
    # Load dữ liệu
    data, labels, pos_neg_labels = LoadData_sentiment('full_data_sentiment_only.xml')
    print "Number of items in train and test: "
    print pos_neg_labels[:number_in_training_set].count(0)
    print pos_neg_labels[number_in_training_set:].count(0)

    print "Normalize Data..."
   
    data = NormalizeData (data)

    # print "Word2Vec phrase..."
    data_vector = Word2Vec_sentiment(data, labels, pos_neg_labels, number_in_training_set,aspect_size, sentiment_size,w2v_model)

    # Gợi ý phân lớp thông qua mô hình Word2Vec
    print "Word2Vec Sentiment Prediction Suggestion (50%)"
    Word2Vec_classifier_2 (w2v_model,data,[],aspect_size,number_in_training_set)

    # Do dữ liệu bự nên lưu thành file luôn
    file_full_data = open('full_data_sentiment_only_word2vec.txt','w')
    for i in range(len(data_vector)):
        for j in range(len(data_vector[i])):
            file_full_data.write(str(data_vector[i][j]) + '\t')
        file_full_data.write('\n')

    file_label = open('full_labels_sentiment_only_word2vec.txt','w')
    for i in range(len(pos_neg_labels)):
        file_label.write(str(pos_neg_labels[i]) + '\n')

    print "Positive, Negative in train"
    print pos_neg_labels[:number_in_training_set].count(0)
    print pos_neg_labels[:number_in_training_set].count(1)

    return [1,1,1]

if __name__ == '__main__':
    # result = []
    print "Loading Word2Vec pretrained model..."
    w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format('GoogleNews.bin', binary=True)  # C binary format

    test_rbm(w2v_model,learningrate=0.1, training_epochs=50, number_in_training_set = 0,
         aspect_size = 100, sentiment_size = 100,n_hidden=500)
