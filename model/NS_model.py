import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid, BCELossWithSigmoid
from preprocess import *
import pickle

class Negative_Sampling:
    def __init__(self, vocab_size,  projection, lr):
        self.Embedding = Embedding(vocab_size , projection)
        self.N_Embdding = Embedding(vocab_size , projection)
        self.lr = lr

        self.layers = [self.Embedding, self.N_Embdding]
        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def forward(self, x, sampled):
        '''
        x = (N, 1) Batch x 1
        sampled = (N, sampled(k) * skip_size + 1)
        '''
        self.x = x
        #N x projection
        self.hidden = self.Embedding.forward(x)
        #N x 1 x projection
        out = np.expand_dims(self.hidden, axis = 1)
        #N x sampled x projection
        self.vec = self.N_Embdding.forward(sampled)
        #N x sampled
        output = np.sum(out * self.vec, axis = 2)
        return output

    def backward(self, dout):
        #dout(N x s)

        #d_emb ==> (S,D)
        d_nemb = np.matmul(dout.T, self.hidden)
        #d_emb ==> (sample, proj)
        dout = np.expand_dims(dout, axis = 2)
        d_emb = np.sum(dout * self.vec, axis = 1)

        self.N_Embdding.backward(d_nemb, self.lr)
        self.Embedding.backward(d_emb, self.lr)


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
