#!/usr/bin/python
# -*- coding: utf-8 -*-

import CONSTANT
import timeit
import os
import numpy
import random
from Word2Vec_classifier import Word2Vec_classifier_1
from bs4 import BeautifulSoup
import theano
import theano.tensor as T
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from theano.tensor.shared_randomstreams import RandomStreams
try:
    import PIL.Image as Image
except ImportError:
    import Image

#Định nghĩa biến const toàn cục - các index của tập train - dev và test

# start-snippet-1
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

# Phần cần chú ý để cập nhật ma trận trọng số W  -----------------------------------------

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

def LoadWord2VecLabels(filename):
    file = open(filename,'r')
    data= file.read().splitlines()
    return data

def LoadWord2VecData(filename):
    temp_train_set = []
    with open(filename) as f:
        for line in f:  #Line is a string
            #split the string on whitespace, return a list of numbers
            # (as strings)
            numbers_str = line.split()
            #convert numbers to floats
            numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
            temp_train_set.append(numbers_float)

    return temp_train_set

def Word2Vec_classifier_1_inscript (w2v_model, sentences, number_in_training_set):
    return_vectors = []
    for i in range(len(sentences[number_in_training_set:])): # predict test
        i = i + number_in_training_set
        words = sentences[i].split()
        words_in_vocabs = []
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                words_in_vocabs.append(words[j])
        if (words_in_vocabs != []):
            food_point = w2v_model.n_similarity(words_in_vocabs, ['thức','đồ','ăn','uống','món'])
            staff_point = w2v_model.n_similarity(words_in_vocabs, ['nhân','viên','phục','vụ'])
            ambience_point = w2v_model.n_similarity(words_in_vocabs, ['không','gian','trang','trí'])
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
            if ('FOOD' in training_labels[i][j]):
                label = label + ' 1'
            if ('AMBIENCE' in training_labels[i][j]):
                label = label + ' 3'
            if ('STAFF' in training_labels[i][j]):
                label = label + ' 5'
    return new_labels


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
    return numpy.array(newlabels)

# Hàm chuyển câu thành vector bằng công cụ word2vec
def Word2Vec(data,labels, pos_neg_labels,number_in_training_set,aspect_size, sentiment_size,w2v_model):

    # Chuyển câu thành vector.........
    for i in range(len(data)):
        # Khai báo vector cho 1 câu, số chiều mặc định là 300
        vector = numpy.zeros(300)
        words = data[i].split()
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                vector = vector + w2v_model[words[j]] # Cộng hết tất cả vector của từ lại sẽ tạo thành vector của câu
        if (i==0):
            data_vector = vector
        else:
            data_vector = numpy.vstack((data_vector,vector))

    # Thêm tất cả các câu (train lẫn test) những số 0 vào để đủ slot cho nhãn predict được
    for i in range(len(data)):
        # Tạo vector cần thêm vào
        add_vec = numpy.zeros(2*aspect_size)
        vector2 = numpy.append(data_vector[i],add_vec)
        if (i == 0):
            return_data_vector = vector2
        else:
            return_data_vector = numpy.vstack((return_data_vector,vector2))


    # Vị trí dành cho aspect
    for i in range(len(data[:number_in_training_set])):
        if (labels[i] == 1): # Food
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-aspect_size-1-j] = 1

        elif (labels[i] == 5): # Staff
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-1-j] = 1

        elif (labels[i] == 3): # Ambience
            for j in range(aspect_size):
                return_data_vector[i][return_data_vector.shape[1]-1-j] = 1

    return return_data_vector

# Hàm dùng để tính Precision, Recall và F1 của main_aspect
def Precision_Recall_F1 (a, aspect_size,test_labels):
    result = []
    # Food =========================
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(a)):
        food = sum(a[i][len(a[i])-3*aspect_size:len(a[i])-2*aspect_size])
        staff = sum(a[i][len(a[i])-2*aspect_size:len(a[i])-aspect_size])
        ambience = sum(a[i][len(a[i])-aspect_size:len(a[i])])

        if (food >= staff and food >= ambience  and test_labels[i] == 1):
            true_positive += 1
        elif (food >= staff and food >= ambience  and test_labels[i] != 1) :
            false_positive += 1
        elif ((food < staff or food < ambience)  and test_labels[i] == 1) :
            false_negative += 1
    print "For Food Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)
    print f1
    print "================================="
    food_result = [pre,recall,f1]
    result.append(food_result)

    # Staff =========================
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(a)):
        food = sum(a[i][len(a[i])-3*aspect_size:len(a[i])-2*aspect_size])
        staff = sum(a[i][len(a[i])-2*aspect_size:len(a[i])-aspect_size])
        ambience = sum(a[i][len(a[i])-aspect_size:len(a[i])])

        if (staff >= food and staff >= ambience  and test_labels[i] == 5):
            true_positive += 1
        elif (staff >= food and staff >= ambience and test_labels[i] != 5) :
            false_positive += 1
        elif ((staff < food or staff < ambience)  and test_labels[i] == 5) :
            false_negative += 1
    print "For Staff Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)
    print f1
    print "================================="
    staff_result = [pre,recall,f1]
    result.append(staff_result)

    # Ambience =========================
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(a)):
        food = sum(a[i][len(a[i])-3*aspect_size:len(a[i])-2*aspect_size])
        staff = sum(a[i][len(a[i])-2*aspect_size:len(a[i])-aspect_size])
        ambience = sum(a[i][len(a[i])-aspect_size:len(a[i])])

        if (ambience >= food and ambience >= staff  and test_labels[i] == 3):
            true_positive += 1
        elif (ambience >= food and ambience >= staff  and test_labels[i] != 3) :
            false_positive += 1
        elif ((ambience < food or ambience < staff)  and test_labels[i] == 3) :
            false_negative += 1
    print "For Ambience Aspect ================="
    print 'Precision is: '
    pre = 0
    if (true_positive+false_positive != 0):
        pre = true_positive*100*1.0/(true_positive+false_positive)
    print pre

    print 'Recall is: '
    recall = 0
    if ((true_positive+false_negative) != 0):
        recall = true_positive*100*1.0/(true_positive+false_negative)
    print recall

    print 'F1 is: '
    f1 = 0
    if ((pre+recall) != 0):
        f1 = 2*pre*recall*1.0/(pre+recall)
    print f1
    print "================================="
    am_result = [pre,recall,f1]
    result.append(am_result)

    return result
def Word2Vec_aspect(data,labels,number_in_training_set, aspect_size,w2v_model):

    # Chuyển câu thành vector.........
    for i in range(len(data)):
        # Khai báo vector cho 1 câu, số chiều mặc định là 300
        vector = numpy.zeros(300)
        words = data[i].split()
        for j in range(len(words)):
            if words[j] in w2v_model.vocab: # Thử xem trong mô hình có từ đó không
                vector = vector + w2v_model[words[j]] # Cộng hết tất cả vector của từ lại sẽ tạo thành vector của câu
        if (i==0):
            data_vector = vector
        else:
            data_vector = numpy.vstack((data_vector,vector))

    # Thêm tất cả các câu (train lẫn test) những số 0 vào để đủ slot cho sentiment
    for i in range(len(data)):
        # Tạo vector cần thêm vào
        add_vec = numpy.zeros(3*aspect_size)
        vector2 = numpy.append(data_vector[i],add_vec)
        if (i == 0):
            return_data_vector_aspect = vector2
        else:
            return_data_vector_aspect = numpy.vstack((return_data_vector_aspect,vector2))


    # Vị trí dành cho aspect
    for i in range(len(data[:number_in_training_set])):
        if (labels[i] == 1): # FOOD
            for j in range(aspect_size):
                return_data_vector_aspect[i][return_data_vector_aspect.shape[1]-2*aspect_size-1-j] = 1

        elif (labels[i] == 5): # STAFF
            for j in range(aspect_size):
                return_data_vector_aspect[i][return_data_vector_aspect.shape[1]-aspect_size-1-j] = 1

        elif (labels[i] == 3): # AMBIENCE
            for j in range(aspect_size):
                return_data_vector_aspect[i][return_data_vector_aspect.shape[1]-1-j] = 1
    return return_data_vector_aspect


# Hàm dùng để tính Precision, Recall và F1 của main_aspect
def Precision_Recall_F1_sentiment (a, sentiment_size,test_labels):
    result = []
    # Food =========================
    true = 0
    for i in range(len(a)):
        positive = sum(a[i][len(a[i])-2*sentiment_size:len(a[i])-sentiment_size])
        negative = sum(a[i][len(a[i])-sentiment_size:len(a[i])])

        if (positive >= negative  and test_labels[i] == '0'):
            true += 1
        elif (negative >= positive  and test_labels[i] == '1') :
            true += 1
    result = true*100.0/len(a)
    return result

# Hàm chạy predict đợt 1 (predict suggestion) của mô hình word2vec
def Word2Vec_Predict_Suggestion(test_vector,aspect_size):
    file = open('word2vec_predict_suggestion.txt','r')
    word2vec_predict = file.read().splitlines()
    for i in range(len(test_vector)):
        if(word2vec_predict[i] == '1'):
            for j in range(aspect_size/2):
                test_vector[i][len(test_vector[i])-2*aspect_size-1-j] = 1
        if(word2vec_predict[i] == '5'):
            for j in range(aspect_size/2):
                test_vector[i][len(test_vector[i])-aspect_size-1-j] = 1
        if(word2vec_predict[i] == '3'):
            for j in range(aspect_size/2):
                test_vector[i][len(test_vector[i])-1-j] = 1
    return test_vector
def Word2Vec_Predict_Suggestion_sentiment(test_vector,sentiment_size):
    file = open('word2vec_sentiment_predict_suggestion.txt','r')
    word2vec_predict = file.read().splitlines()
    for i in range(len(test_vector)):
        if(word2vec_predict[i] == '0'):
            for j in range(sentiment_size/2):
                test_vector[i][len(test_vector[i])-sentiment_size-1-j] = 1
        if(word2vec_predict[i] == '1'):
            for j in range(sentiment_size/2):
                test_vector[i][len(test_vector[i])-1-j] = 1

    return test_vector


# Hàm main
def test_rbm(data_vector,labels,learning_rate=0.1, training_epochs=20, number_in_training_set = 150,
              batch_size=20,aspect_size = 100, sentiment_size = 100,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):

    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    Data sets bao gồm 3 thành phần được thể hiện theo cây
    # + train set
    #     - ma trận đầu vào (input), mỗi dòng là một sample, gồm 784 cột tương ứng với 784 pixels trong ảnh
    #     - vector nhãn (target) có số chiều bằng số mẫu
    # + valid set (tương tự trên)
    # + test set (tương tự trên)
    # Mỗi set đều ở dạng floatX để có thể chạy trên GPU, và có thêm một clone của nó dưới dạng integer
    # Quy định nhãn dưới dạng số:
    # 0. Restaurant
    # 1. Food
    # 2. Drinks
    # 3. Ambience
    # 4. Prices
    # 5. Service
    # 6. Other aspect
    # Còn đối với nhãn sentiment thì quy định nhu sau:
    # 0. positive
    # 1. negative
    # 2. cả hai
    """
    # print "Loading Data..."
    # # Load dữ liệu
    # data, labels, pos_neg_labels = LoadData('full_data_aspect_only.xml')
    #
    #
    # print "Normalize Data..."
    # # Chuẩn hóa reviews: lower case, tách từ, loại bỏ stopword, không cần bỏ thời gian để bỏ đi dấu chấm vì khi dùng thư viện
    # # để tách từ nó sẽ tự động loại những cái đó ra
    # data = NormalizeData (data)
    #
    #
    # print "Word2Vec phrase..."
    # data_vector = Word2Vec(data, labels, pos_neg_labels, number_in_training_set,aspect_size, sentiment_size,w2v_model)

    # Loading data

    temp_train_set = []
    with open('Aspect_train_set.TXT') as f:
        for line in f:  #Line is a string
            #split the string on whitespace, return a list of numbers
            # (as strings)
            numbers_str = line.split()
            #convert numbers to floats
            numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
            temp_train_set.append(numbers_float)



    # training_set =  data_vector[:7700] + data_vector[38694:46394] + data_vector[52876:60576]
    # training_labels =  labels[:7700] + labels[38694:46394] + labels[52876:60576]
    print "Make shared train set"
    # Training set x là số float, training set y là số integer
    train_set_x = theano.shared(numpy.asarray(temp_train_set,dtype=theano.config.floatX),borrow=True)
    # train_set_y = theano.shared(numpy.asarray(data_vector_train,dtype=theano.config.floatX),borrow=True)

    # print labels.count('1')
    # print labels.count('5')
    # print labels.count('3')



    # Tính số lượng vòng lặp cần để chạy hết tất cả các mẫu, vì ta chia ra nhiều batch,
    # batch_size là số lượng mẫu trong một batch, định sẵn là 20 ảnh
    # n_train_batches là số vòng lặp để chạy hết tất cả các batches, tính được cần phải có 2500 vòng lặp
    # train_set_x.get_value(borrow=True).shape[0] chính là tổng số ảnh dùng để huấn luyện (gồm 50000 ảnh)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # Định nghĩa các symbolic variables cho dữ liệu
    index = T.lscalar()    # index của [mini]batch, dùng long integer scalar
    x = T.matrix('x')  # dữ liệu được thể hiện dưới dạng ma trận ảnh, dùng để feed vào mô hình RBM ngay dòng lệnh intial

    # Khai báo biến random
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # Cài đặt vùng nhớ cho persistent chain, là ma trận khởi tạo bằng ma trận 0 với số chiều là
    # batch_size x n_hidden (tức là 20 x 500 theo mặc định)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # Tạo biến RBM để train, test... số node visible chính là tổng số vocabulary + 3 nhãn nữa
    # Lúc này chỉ mới dùng symbolic x để biểu diễn input thôi, chứ chưa phải input thật sự
    rbm = RBM(input=x, n_visible= 300 +3*aspect_size,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # Lấy cost và update qua 1 bước gọi hàm CD với k = 15, khi này cost và update bản thân nó không phải là số
    # nó vẫn chỉ là symbolic do self.input là symbolic x, không phải là trainings sample thật sự, nó là một đống
    # symbolic expression thôi, để một hồi đưa vào theano.function để chạy
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=1)

    #################################
    #     Training the RBM          #
    #################################
    # Tạo thư mục để chứa ảnh plot ra được, sau đó chuyển current working directory qua thư mục đó

    # Tạo hàm huấn luyện
    # Theano function có thể không trả về output, mục tiêu chính của cái train_rbm này là để cho nó update trọng số
    # parameters của RBM thôi
    # Input: index của cái batch
    # Output: không có
    # Updates: là từ điển update
    # Givens: givens cho biết những thứ đã có, vì trong cái symbolic expression cost (nó được trả về bởi hàm rbm.get_cost_updates)
    # nó có chứa chữ x (tại vì mình đã truyền self.input = x ở phía trên, trong khi x vẫn là một symbolic exp chứ không
    # phải là một giá trị nhất định nào đó. Vì vậy chỗ này mình phải cho cái máy biết x là gì
    # x chính là tập train_set_x (ở dạng số floatX), với số mẫu chạy từ batch này tới batch liền kế nó, tức là mỗi lần
    # gọi hàm train_rbm này, ta chỉ chạy trên 1 batch gồm n mẫu nhất định thôi
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()
    print "Training model..."
    # Bắt đầu chạy từng vòng lặp, training epochs mặc định là 15 vòng
    for epoch in xrange(training_epochs):
        # Xét từng tập train (tức từng tập mini batch)
        mean_cost = []
        # xrange không tạo ra một list rồi lưu vào bộ nhớ giống như lệnh range, mà nó cho biến batch_index nhích lên từ từ
        # n_train_batches gồm 2500 số vòng lặp (đã tính phía trên), tức batch_index sẽ chạy từ từ trong đó (từ 0 đến 2499)
        # Mỗi lần lặp là nó sẽ chạy trên mẫu dữ liệu trên 1 batch, gồm 20 ảnh
        for batch_index in xrange(n_train_batches):

            mean_cost += [train_rbm(batch_index)]


    end_time = timeit.default_timer()
    # Kết thúc quá trình huấn luyện

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))


    ###########################################################
    #    QUÁ TRÌNH PREDICTION SUGGESTION CỦA WORD2VEC MODEL   #
    ###########################################################

    # Gợi ý phân lớp thông qua mô hình Word2Vec
    test_vectors = data_vector[number_in_training_set:]
    test_vectors_after = Word2Vec_Predict_Suggestion(test_vectors,aspect_size)

    # Gán theano
    test_set_x = theano.shared(numpy.asarray(test_vectors_after,dtype=theano.config.floatX),borrow=True)
    test_labels = labels[number_in_training_set:]

    #################################1
    #    Lấy mẫu từ mô hình RBM     #
    #################################
    # Lấy ra số lượng mẫu test
    # number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    #
    # # Lấy random số mẫu trong số mẫu test trên rồi gán chúng vào persistent chain
    # test_idx = rng.randint(number_of_test_samples - n_chains)
    # persistent_vis_chain = theano.shared(
    #     numpy.asarray(
    #         test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
    #         dtype=theano.config.floatX
    #     )
    # )
    # test_set_x = train_set_x
    # test_labels = train_labels

    plot_every = 1
    # Lấy mẫu Gibbs - định nghĩa hàm lặp số lần là giá trị của biến plot_every
    # Lấy giá trị visible sau khi lặp nhiêu đó lần rồi trả về

    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, test_set_x],
        n_steps=plot_every
    )

    # Thêm vào từ điển cái biến shared để dành chỗ cho peristent chain
    # Cái từ điển update này dùng để lấy giá trị mới nhất của cái vòng lặp trên
    # Nếu không có nó coi như giá trị mới tính được phía trên luôn giữ nguyên không đổi,
    # Nếu không có nó có thể dãy số random được giống nhau hoàn toàn trong các vòng lặp
    # Link tham khảo: Kéo xuống giữa: http://deeplearning.net/software/theano/library/scan.html
    # Phần Using shared variables - Gibbs sampling
    updates.update({test_set_x: vis_samples[-1]})

    # Xây dựng cái hàm dựa vào vòng lặp scan phía trên
    # Cái scan phía trên là expression, expression ở trên tiếp tục được nhúng cái expression phía dưới này,
    # chủ yếu để làm thành cái hàm và một lát được gọi lại để chạy, gán update=updates để các biến random
    # ra khác nhau ở mỗi vòng lặp
    # Mình cần có mean field (giá trị sau sigmoid) để plot hình và cái vis sample để reinit cái persistent chain

    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    vis_mf, vis_sample = sample_fn()

    ##########################################
    #    QUÁ TRÌNH TÍNH PRECISION RECALL F1  #
    ##########################################
    a = vis_sample

    result = Precision_Recall_F1 (a, aspect_size,test_labels)


    # ======================================== . . =============================================
    # # Lấy mẫu những từ của aspect food
    # food_vis_input = []
    # sample = [0] * (len(vocabulary)+50)
    # food_vis_input.append(sample)
    # for i in range(len(food_vis_input)):
    #     for j in range(10):
    #         food_vis_input[i][len(food_vis_input[i])-21-j] = 1
    # test_generate = theano.shared(numpy.asarray(food_vis_input,dtype=theano.config.floatX),borrow=True)
    # (
    #     [
    #         presig_hids,
    #         hid_mfs,
    #         hid_samples,
    #         presig_vis,
    #         vis_mfs,
    #         vis_samples
    #     ],
    #     updates
    # ) = theano.scan(
    #     rbm.gibbs_vhv,
    #     outputs_info=[None, None, None, None, None, test_generate],
    #     n_steps=40
    # )
    #
    # # Thêm vào từ điển cái biến shared để dành chỗ cho peristent chain
    # # Cái từ điển update này dùng để lấy giá trị mới nhất của cái vòng lặp trên
    # # Nếu không có nó coi như giá trị mới tính được phía trên luôn giữ nguyên không đổi,
    # # Nếu không có nó có thể dãy số random được giống nhau hoàn toàn trong các vòng lặp
    # # Link tham khảo: Kéo xuống giữa: http://deeplearning.net/software/theano/library/scan.html
    # # Phần Using shared variables - Gibbs sampling
    # updates.update({test_generate: vis_samples[-1]})
    #
    # sample_fn = theano.function(
    #     [],
    #     [
    #         vis_mfs[-1],
    #         vis_samples[-1]
    #     ],
    #     updates=updates,
    #     name='sample_fn'
    # )
    #
    # vis_mf, vis_sample = sample_fn()
    #
    # a = vis_sample
    # for i in range(len(vocabulary)):
    #     if (a[0][i] == 1):
    #         print vocabulary[i]

    return result
if __name__ == '__main__':

    print "Loading Data..."
    # Load dữ liệu
    data, labels, pos_neg_labels = LoadData(CONSTANT.DATASET_FOLDER_DIR + '/' + CONSTANT.Output_FSA)
    print "Normalize Data..."
    # Chuẩn hóa reviews: lower case, tách từ, loại bỏ stopword, không cần bỏ thời gian để bỏ đi dấu chấm vì khi dùng thư viện
    # để tách từ nó sẽ tự động loại những cái đó ra
    data = NormalizeData (data)

    # Load model lên
    w2v_model = gensim.models.word2vec.Word2Vec.load(CONSTANT.DATASET_FOLDER_DIR+'/'+CONSTANT.Word2Vec_ENG_model)

    # print "Word2Vec phrase..."
    data_vector = Word2Vec_aspect(data,labels,150, 100,w2v_model)

    # Gợi ý phân lớp thông qua mô hình Word2Vec
    print "Word2Vec Aspect Prediction Suggestion (50%)"
    Word2Vec_classifier_1_inscript (w2v_model,data,150)

    # Do dữ liệu bự nên lưu thành file luôn
    file_full_data = open('full_data_aspect_only_word2vec.txt','w')
    for i in range(len(data_vector)):
        for j in range(len(data_vector[i])):
            file_full_data.write(str(data_vector[i][j]) + '\t')
        file_full_data.write('\n')

    file_label = open('full_labels_aspect_only_word2vec.txt','w')
    for i in range(len(labels)):
        file_label.write(str(labels[i]) + '\n')

    result = []
    print "Loading Word 2 Vec already data"
    # data_vector = LoadWord2VecData('full_data_aspect_only_word2vec.txt')
    # labels = LoadWord2VecLabels('full_labels_aspect_only_word2vec.txt')


    for i in range(1):
        result.append(test_rbm(data_vector,labels,learning_rate=0.1, training_epochs=3, number_in_training_set = 150,
         batch_size=4000,aspect_size = 100, sentiment_size = 100,
         n_chains=20, n_samples=10, output_folder='rbm_plots',
         n_hidden=300))
    print result

    file_result = open("result_RBM_Word2Vec_sentiment.txt",'w')
    for i in range(len(result)):
        file_result.write(str(result[i])+'\n')
