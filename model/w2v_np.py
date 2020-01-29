import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid
from preprocess import *
import pickle

class BCELossWithSigmoid:
    def __init__(self):
        self.params = None
        self.grads = None
        self.eps = 1e-7

        self.y_pred , self.target = None, None
        self.loss = None

    def forward(self, y_pred, target):

        self.target = target
        self.y_pred = sigmoid(y_pred)

        number = target.shape[0]

        self.loss = -self.target * np.log(self.y_pred + self.eps) - (1 - self.target) * np.log(1 - self.y_pred + self.eps)

        self.loss = np.sum(self.loss) / number

        return self.loss

    def backward(self):
        dx = self.y_pred - self.target
        return dx

class HS_skipgram:
    def __init__(self, vocab_size, projection, lr):
        self.Embedding = Embedding(vocab_size, projection)
        self.HSvector = Embedding(vocab_size - 1 , projection)
        self.lr = lr
        self.layers = [self.Embedding, self.HSvector]

        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x, label):
        '''
        inputs : 1 x D(projection)
        label : 1 x [direction_path(1, depth), idx_path(1, depth)]
        '''

        dir_path = np.expand_dims(label[0],1)
        idx_path = label[1]

        self.x = x
        
        self.hidden = self.Embedding.forward(self.x)

        self.hirearchy_vectors = self.HSvector.forward(idx_path)

        out = np.sum(self.hirearchy_vectors * self.hidden, axis = 1, keepdims= True)

        return out, dir_path

    def backward(self, dout):

        W_in, W_out = self.params
        #truth length x hidden
        d_lin = np.matmul(dout , self.hidden)
        #d_h = np.matmul(dout.T, self.hirearchy_vectors)
        d_h = np.sum(dout * self.hirearchy_vectors, axis = 0)

        self.HSvector.backward(d_lin, self.lr)
        self.Embedding.backward(d_h, self.lr)

        '''
        print((self.grads[0] == self.Embedding.grads[0]).all())
        print((self.grads[1] == self.HSvector.grads[0]).all())
        '''
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

    def query(self, word, word2idx, idx2word, top = 5):

        if word not in word2idx:
            print("%s는 corpus 안에 존재하지 않습니다"%word)
            return
        
        W_in , _ = self.params

        query_id = word2idx[word]
        query_vec = W_in[query_id]

        #오름차순에 의해 정렬
        similarity = cosine_similarity(W_in , query_vec)

        #자기 자신 제외
        result = similarity.argsort()[-top-1:-1]

        print(word)

        for i in range(top):
            print(idx2word[int(result[i])] , similarity[int(result[i])])

class Negative_Sampling:
    def __init__(self, vocab_size, projection):
        self.Embedding = Embedding(vocab_size, projection)
        self.N_Embdding = Embedding(vocab_size , projection)

        self.layers = [self.Embedding, self.N_Embdding]
        self.params = []
        self.grads = []

        for layer in layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def forward(self, x):
        print(0)