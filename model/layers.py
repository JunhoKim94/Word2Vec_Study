import numpy as np

'''
MAX_EXP = 7
EXP_TABLE = 1000

exp = np.array([i for i in range(EXP_TABLE)])
exp = exp / EXP_TABLE / 2
exp -= 1
exp *= MAX_EXP

exp = np.exp(exp)
exp = exp / (1 + exp)

def sigmoid(x):
    return exp[int(x + MAX_EXP) * (EXP_TABLE / MAX_EXP / 2)]

'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

        self.loss = -1 * self.target * np.log(self.y_pred + self.eps) - (1 - self.target) * np.log(1 - self.y_pred + self.eps)

        self.loss = np.sum(self.loss) / number

        return self.loss

    def backward(self):
        dx = self.y_pred - self.target
        return dx


class Embedding:
    def __init__(self, input_size, output_size):

        #self.W = np.random.uniform(size = (self.input_size, self.output_size))
        self.params = [np.random.uniform(size = (input_size, output_size)).astype(np.float32)]
        self.grads = [np.zeros_like(self.params[0])]

    def forward(self, x):
        '''
        x = list or array
        '''
        self.idx = x
        W, = self.params
        output = W[self.idx]

        return output

    def backward(self, dout, lr):
        '''
        idx 해당 하는 w 만 grad = 1 * dout
        나머지 0
        '''
        #dW, = self.grads
        W,  = self.params
        #dW[self.idx] += dout
        W[self.idx] -= dout * lr
        #np.add.at(dW, self.idx, dout)

    def _zero_grad(self):
        dW, = self.grads
        dW[...] = 0


class Linear:
    def __init__(self, W):

        self.grad = [np.zeros_like(W)]
        self.params = [W]

    def forward(self, x):
        '''
        W = (D,H)
        x = (N,D)

        out : (N,D)
        '''

        W, = self.params

        self.x = x
        output = np.matmul(self.x,W)

        return output

    def backward(self, dout):
        '''
        input: d_out (N,H)
        self.x : (N,D)
        output: dW : (D,H)
                db : (H,)
        '''
        W,b = self.paramss

        dx = np.dot(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grad[0][...] = dW
        self.grad[1][...] = db

        return dx

def softmax(z):
    #numerically stable softmax
    z = z - np.max(z, axis =1 , keepdims= True)
    _exp = np.exp(z)
    _sum = np.sum(_exp,axis = 1, keepdims= True)
    sm = _exp / _sum

    return sm

def cross_entropy_loss(z, target):
    if z.ndim == 1:
        target = target.reshape(1, target.size)
        z = z.reshape(1, z.size)

    if z.size == target.size:
        target = target.argmax(axis = 1)

    batch_size = z.shape[0]

    return -np.sum(np.log(z[np.arange(batch_size), t] + 1e-7)) / batch_size


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        '''
        input: d_out (N,H)
        self.x : (N,H)
        output: 
        '''
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

