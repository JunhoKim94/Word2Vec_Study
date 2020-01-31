import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid, BCELossWithSigmoid
from preprocess import *
import pickle

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

    def forward(self, x, idx_path):
        '''
        inputs : 1 x D(projection)
        label : 1 x [direction_path(1, depth), idx_path(1, depth)]
        '''

        self.x = x
        
        self.hidden = self.Embedding.forward(self.x)

        self.hirearchy_vectors = self.HSvector.forward(idx_path)

        out = np.sum(self.hirearchy_vectors * self.hidden, axis = 1, keepdims= True)

        return out

    def backward(self, dout):

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


    def load(self, path):
        with open("./bestmodel.pickle", 'rb') as f:
            x = pickle.load(f)

        self.params = x
        for param,layer in zip(self.params, self.layers):
            layer.params = [param]

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
