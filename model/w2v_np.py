import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear
from utils import _Huffman_Tree

class BCELoss:
    def __init__(self):
        self.params = None
        self.grads = None
        self.eps = 1e-8

        self.y_pred , self.target = None, None
        self.loss = None

    def forward(self, y_pred, target, dim = 1):

        self.y_pred = y_pred
        self.target = target
        self.dim = dim
        number = target.shape[0]
        
        self.loss = -self.target * np.log(self.y_pred + self.eps) - (1 - self.target) * np.log(1 - self.y_pred + self.eps)
        self.loss = np.sum(self.loss, axis = dim) / number

        return self.loss

    def backward(self, dout = 1):
        dx = (self.y_pred - self.target) / (self.y_pred * (1 - self.y_pred) +  self.eps) 
        return dx * dout

class Hsoftmax:
    def __init__(self, vocab_size, projection, sample_size):
        self.Embedding = Embedding(vocab_size, projection)
        self.HSvector = Embedding(vocab_size - 1 , projection)
        self.sigmoid = Sigmoid()
        self.sample_size = sample_size

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
        label 과 output 의 argmax를 비교해서 같으면 1 틀리면 0 을 부여한 후 이를 target vector로 설정해야됨
        '''

        dir_path = np.array(label[0])
        idx_path = np.expand_dims(label[1], 1)
        self.x = x
        
        self.hidden = self.Embedding.forward(x)

        self.hirearchy_vectors = self.HSvector.forward(dir_path)

        out = np.matmul(self.hirearchy_vectors , self.hidden.T )

        out = self.sigmoid.forward(out)

        mask = np.zeros_like(out)
        mask[mask >= 0.5] = 1

        target = np.zeros_like(out)
        target[mask == idx_path] = 1

        return out , target

    def backward(self, dout):

        W_in, W_out = self.params
        #length x 1
        d_sig = self.sigmoid.backward(dout)
        #vocab -1 x hidden
        d_lin = np.matmul(d_sig , self.hidden)
        d_h = np.matmul(dout.T, self.hirearchy_vectors)

        self.HSvector.backward(d_lin)
        self.Embedding.backward(d_h)

        '''
        print((self.grads[0] == self.Embedding.grads[0]).all())
        print((self.grads[1] == self.HSvector.grads[0]).all())
        '''