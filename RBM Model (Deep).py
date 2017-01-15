#!/usr/bin/python
# -*- coding: utf-8 -*-

import timeit
import os
import numpy
from bs4 import BeautifulSoup
import theano
import theano.tensor as T
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from theano.tensor.shared_randomstreams import RandomStreams
try:
    import PIL.Image as Image
except ImportError:
    import Image


train_food_lim = 600
dev_food_lim = 800
test_food_lim = 1000

train_staff_lim = 60
dev_staff_lim = 80
test_staff_lim = 100

train_ambience_lim = 60
dev_ambience_lim = 80
test_ambience_lim = 100

# Định nghĩa lớp RBM
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


        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

# Hàm load dữ liệu
def LoadData (filename):
    file = open(filename,'r')
    data= file.read().splitlines()
    return data

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

# Tạo tập vocabulary
def MakeVocabSet (train_data_food, train_data_staff, train_data_ambience):
    vocabulary = []
    new_training_data = train_data_food[:train_food_lim]  + train_data_staff[:train_staff_lim] + train_data_ambience[:train_ambience_lim]
    print 'Tokenize new words in training data'

    # XÂy dựng tập từ vựng
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(new_training_data)):
        words = tokenizer.tokenize(new_training_data[i])
        vocabulary = vocabulary + words

    # Loại bỏ những từ giống nhau cho tập vocab
    vocabulary = list(set(vocabulary))
    print '- Vocabulary set completed'

    del new_training_data
    del words

    return vocabulary


def Convert2VocabVector (data, last_limit, vocabulary):
    # Biến trả về
    new_vectors =[]
    tokenizer = RegexpTokenizer(r'\w+')

    # Chuyển training thành vector số cho mỗi loại aspects
    for i in range(len(data[:last_limit])):
        wordsvector = [0] * (len(vocabulary)+3) # Thêm 3 để dành cho nhãn
        words = tokenizer.tokenize(data[i])
        for k in range(len(words)):
            if (words[k] in vocabulary):
                index = vocabulary.index(words[k])
                wordsvector[index] += 1
        new_vectors.append(wordsvector)

    del wordsvector
    del words
    
    return new_vectors


def Convert2Np (labels,number_of_hidden_nodes):
    newlabels = []
    for i in range(len(labels)):
        np_label = [0]*number_of_hidden_nodes
        for j in range(number_of_hidden_nodes):
            if(str(j) in labels[i]):
                np_label[j] = 1
        newlabels.append(np_label)
    return numpy.array(newlabels)

# Hàm gán thẳng labels và data vector
def EmbedLabels(data_vector_food, data_vector_staff, data_vector_ambience,
                train_labels_food, train_labels_staff, train_labels_ambience):
    data_vector_train = data_vector_food[:train_food_lim] + data_vector_staff[:train_staff_lim] + data_vector_ambience[:train_ambience_lim]
    data_vector_dev = data_vector_food[train_food_lim:dev_food_lim] + data_vector_staff[train_staff_lim:dev_staff_lim] + data_vector_ambience[train_ambience_lim:dev_ambience_lim]
    data_vector_test = data_vector_food[dev_food_lim:test_food_lim] + data_vector_staff[dev_staff_lim:test_staff_lim] + data_vector_ambience[dev_ambience_lim:test_ambience_lim]

    train_labels = train_labels_food[:train_food_lim] + train_labels_staff[:train_staff_lim] + train_labels_ambience[:train_ambience_lim]
    dev_labels = train_labels_food[train_food_lim:dev_food_lim] + train_labels_staff[train_staff_lim:dev_staff_lim] + train_labels_ambience[train_ambience_lim:dev_ambience_lim]
    test_labels = train_labels_food[dev_food_lim:test_food_lim] + train_labels_staff[dev_staff_lim:test_staff_lim] + train_labels_ambience[dev_ambience_lim:test_ambience_lim]

    for i in range(len(data_vector_train)):
        if (train_labels[i] == 1): # Food
            data_vector_train[i][len(data_vector_train[i])-3] = 1
        if (train_labels[i] == 5): # Staff
            data_vector_train[i][len(data_vector_train[i])-2] = 1
        if (train_labels[i] == 3): # Ambience
            data_vector_train[i][len(data_vector_train[i])-1] = 1

    # 2 tập còn lại không nhúng nhãn, chừa slot số 0 để nó tự điền vào (khúc chừa slot là nó nằm trên hàm lúc convert qua vector đã thực hiện rồi)

    return data_vector_train, data_vector_dev, data_vector_test, train_labels, dev_labels, test_labels

# Hàm main
def test_rbm(learning_rate=0.1, training_epochs=40,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):

    print "Loading Data..."
    # Load dữ liệu
    data_food = LoadData('../Training Data/Food_reviews.txt')
    data_staff = LoadData('../Training Data/Staff_reviews.txt')
    data_ambience = LoadData('../Training Data/Ambience_reviews.txt')

    labels_food = [1] * len(data_food)
    labels_staff = [5] * len(data_staff)
    labels_ambience = [3] * len(data_ambience)

    print "Normalize Data..."

    data_food = NormalizeData (data_food)
    data_staff = NormalizeData (data_staff)
    data_ambience = NormalizeData (data_ambience)


    print "Making Vocabulary Set..."

    vocabulary = MakeVocabSet(data_food, data_staff, data_ambience)
    print 'Number of Vocabs: '  + str(len(vocabulary))
    print 'Convert food data\n'
    data_vector_food = Convert2VocabVector (data_food, test_food_lim, vocabulary)

    print 'Convert staff data\n'
    data_vector_staff = Convert2VocabVector (data_staff, test_staff_lim, vocabulary)

    print 'Convert ambience data\n'
    data_vector_ambience = Convert2VocabVector (data_ambience, test_ambience_lim, vocabulary)


    print "Embed labels in to data..."
  
    data_vector_train, data_vector_dev, data_vector_test, train_labels, dev_labels, test_labels \
        = EmbedLabels(data_vector_food, data_vector_staff, data_vector_ambience,labels_food, labels_staff, labels_ambience)


    print "Make shared train set"
    # Training set x là số float
    train_set_x = theano.shared(numpy.asarray(data_vector_train,dtype=theano.config.floatX),borrow=True)


    print train_set_x.eval()


    test_set_x = theano.shared(numpy.asarray(data_vector_dev,dtype=theano.config.floatX),borrow=True)


    count = 0
    for i in range(len(data_vector_train)):
        if(data_vector_train[i][len(data_vector_train[i])-3] == 1):
            count +=1
    print count

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # Định nghĩa các symbolic variables cho dữ liệu
    index = T.lscalar()    # index của [mini]batch, dùng long integer scalar
    x = T.matrix('x')  # dữ liệu được thể hiện dưới dạng ma trận ảnh, dùng để feed vào mô hình RBM ngay dòng lệnh intial

    # Khai báo biến random
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RBM(input=x, n_visible=len(vocabulary) + 3,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

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


    for epoch in xrange(training_epochs):

        mean_cost = []

        for batch_index in xrange(n_train_batches):

            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = timeit.default_timer()
    # Kết thúc quá trình huấn luyện

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    test_set_x = train_set_x
    test_labels = train_labels

    plot_every = 1


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

    
    updates.update({test_set_x: vis_samples[-1]})

    

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

    a = vis_sample

    print a[1:5]
    print 'Predict '
    countfood = 0
    countstaff = 0
    countambience = 0
    count_fs = 0
    count_fa = 0
    count_sa = 0
    countall = 0

    for i in range(len(a)):
        if(a[i][len(a[i])-3] == 1):
            countfood+=1
        if(a[i][len(a[i])-2] == 1.0):
            countstaff+=1
        if(a[i][len(a[i])-1] == 1.0):
            countambience+=1
        if(a[i][len(a[i])-3] == 1 and a[i][len(a[i])-2] == 1):
            count_fs+=1
        if(a[i][len(a[i])-3] == 1 and a[i][len(a[i])-1] == 1):
            count_fa+=1
        if(a[i][len(a[i])-2] == 1 and a[i][len(a[i])-1] == 1):
            count_sa+=1
        if(a[i][len(a[i])-3] == 1 and a[i][len(a[i])-2] == 1 and a[i][len(a[i])-1] == 1):
            countall+=1

    print countfood
    print countstaff
    print countambience
    print count_fs
    print count_fa
    print count_sa
    print countall


    ratio = 0
    for i in range(len(a)):
        if(a[i][len(a[i])-3] == 1 and test_labels[i] == 1):
            ratio += 1
        if(a[i][len(a[i])-2] == 1 and test_labels[i] == 5):
            ratio += 1
        if(a[i][len(a[i])-1] == 1 and test_labels[i] == 3):
            ratio += 1

    print 'Accuracy is: '
    print ratio*100/(1.0*len(a))

if __name__ == '__main__':
    test_rbm()
