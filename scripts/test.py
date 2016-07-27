# Ensure python 3 forward compatibility
from __future__ import print_function
from theano import pp
import numpy as np
import theano
import theano.tensor.nnet as nnet
# By convention, the tensor submodule is loaded as T
import theano.tensor as T

x = T.dvector('x')
y = T.dvector('y')

def layer(x,w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    h = T.tanh(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.1
    return theta - (alpha* T.grad(cost, wrt=theta))

def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)  
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
    return word_vecs, layer1_size

w2v, layer_size = load_bin_vec('../data/NCBI/NCBI_corpus/word_train.bin')

theta1 = theano.shared(np.array(np.random.rand(layer_size + 1, 2*layer_size), dtype = theano.config.floatX))
theta2 = theano.shared(np.array(np.random.rand(2*layer_size + 1, 3), dtype = theano.config.floatX))
hidden_layer = layer(x, theta1)
out = layer(hidden_layer, theta2)
fc = T.sum((out - y)**2)
cost = theano.function(inputs=[x,y],outputs=fc,updates = [(theta1,grad_desc(fc,theta1)),(theta2,grad_desc(fc,theta2))])
run_forward = theano.function(inputs=[x], outputs=out)

file_train = open('../data/NCBI/NCBI_corpus/class_train.txt','r')
cur_cost = 0
j=0
for i in range(10):
    for line in file_train:
        content = line.split(' ')
        if content[1] in w2v:
            exp_y = np.array([0,0,1], dtype=theano.config.floatX)
            if content[0] == 'B':
                exp_y = np.array([1,0,0], dtype=theano.config.floatX)
            elif content[0] == 'I':
                exp_y = np.array([0,1,0], dtype=theano.config.floatX)
            input = w2v[content[1]]
            cur_cost = cost(input, exp_y) 
            j += 1
            if j %50 == 0:
                 print('cost : %s'%(cur_cost,))

file_test = open('../data/NCBI/NCBI_corpus/word_test.txt','r')
right = 0
total = 0
for line in file_train:
    content = line.split(' ')
    y = run_forward(content[1])
    if max(y) == y[0]:
        label = 'B'
    elif max(y) == y[1]:
        label = 'I'
    else:
        label = 'O'
    if label == content[0]:
        right = right + 1
    file_test.write(label + ' ' + content[1])
    total = total + 1
print(right)
print(total)
file_train.close()
file_test.close()
