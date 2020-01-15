import numpy as np


class Embedding:
    def __init__(self, input_size, output_size, padding_idx = None):
        self.input_size = input_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.idx = None

        self.W = np.random.uniform(size = (self.input_size, self.output_size))
        self.grad = [np.zeros_like(self.W)]

    def forward(self, x):
        self.idx = x
        output = self.W[self.idx]

        return output

    def backward(self, dout):
        '''
        idx 해당 하는 w 만 grad = 1 * dout
        나머지 0
        '''
        dW = self.grad
        dW[...] = 0
        np.add.at(dW, self.idx, dout)

class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size


        self.W = np.random.uniform(size = (self.input_size, self.output_size))
        self.b = np.random.uniform(self.output_size)
        self.grad = [np.zeros_like(self.W), np.zeros_like (self.b)]
        self.params = [self.W, self.b]

    def forward(self, x):
        '''
        W = (D,H)
        x = (N,D)

        out : (N,D)
        '''

        self.x = x
        output = np.matmul(self.x,self.W) + self.b

        return output

    def backward(self, dout):
        '''
        input: d_out (N,H)
        self.x : (N,D)
        output: dW : (D,H)
                db : (H,)
        '''
        dx = np.dot(dout, self.W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grad[0][...] = dW
        self.grad[1][...] = db

        return dx

def softmax(self, z):
    #numerically stable softmax
    z = z - np.max(z, axis =1 , keepdims= True)
    _exp = np.exp(z)
    _sum = np.sum(_exp,axis = 1, keepdims= True)
    sm = _exp / _sum

    return sm

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


class Hsoftmax:
    def __init__(self,vocab_size, projection):
        self.vocab_size = vocab_size
        self.projection = projection

        self.HSvector = Embedding(self.vocab_size -1, projection)
        self.sigmoid = Sigmoid()

    def forward(self, x, label):
        '''
        inputs : 1 x [idx]
        label : 1 x [direction_path(2C, depth), idx_path(2C, depth)]
        label 과 output 의 argmax를 비교해서 같으면 1 틀리면 0 을 부여한 후 이를 target vector로 설정해야됨
        ex) output = [0.7, 0.3, 0.4] label = [1, 1, 0]
        --> target = [1, 0, 1]
        
        : BCE Loss 를 사용할 것 : - y_t * log(y_p) - (1-y_t) * log(1 - y_p)
        false --> -log(1 - y_p) = -log(sigmoid(-v_t * h))
        True --> -log(y_p) = -log(sigmoid(v_t*h))

        outputs: list of output & target
        밖에서 loss값 따로 계산해야됨
        '''

        


class HSModel:
    def __init__(self, vocab_size, projection):
        self.vocab_size = vocab_size
        self.projection = projection

        self.embedding = Embedding(self.vocab_size, self.projection)

        self.W = self.embedding.W

    def forward_step(self, x_input, label):
        '''
        skip-gram 중 random sample 하나 학습
        depth < max_depth
        input : x_input (idx, direction_list), idx_path(depth)
        '''

        target_list = []
        output_list = []
        #(1,D) : hidden layer
        proj = self.embedding(x_input)
        
        
        for dir_path, label in label:
            #print(dir_path)

            #(path_length, D)
            hirearchy_vectors = self.HSvector(dir_path)

            #(1, path_length)
            output = np.matmul(proj, hirearchy_vectors.T)
            output = sigmoid(output)
            #print(output.shape)


        return output_list, target_list


    def forward(self, inputs, label):
        '''
        inputs : 1 x [idx]
        label : 1 x [direction_path(2C, depth), idx_path(2C, depth)]
        label 과 output 의 argmax를 비교해서 같으면 1 틀리면 0 을 부여한 후 이를 target vector로 설정해야됨
        ex) output = [0.7, 0.3, 0.4] label = [1, 1, 0]
        --> target = [1, 0, 1]
        
        : BCE Loss 를 사용할 것 : - y_t * log(y_p) - (1-y_t) * log(1 - y_p)
        false --> -log(1 - y_p) = -log(sigmoid(-v_t * h))
        True --> -log(y_p) = -log(sigmoid(v_t*h))

        outputs: list of output & target
        밖에서 loss값 따로 계산해야됨
        '''
        
        output, target = self.forward_step(single_input, single_label)
        proj = self.embedding(x_input)

        #(path_length, D)
        hirearchy_vectors = self.HSvector(dir_path)
        output = np.matmul(proj, hirearchy_vectors.T)
        output = sigmoid(output)




        mask = np.zeros_like(output)
        mask[output >= 0.5] = 1
        #print(label)
        target = np.zeros_like(label)
        target[mask == label] = 1



        return output_list, target_list

    def query(self, word, word2idx, idx2word, top = 5):

        if word not in word2idx:
            print("%s는 corpus 안에 존재하지 않습니다"%word)
            return
        
        for params in self.embedding.parameters():
            self.w = params.data


        query_id = word2idx[word][0]
        query_vec = self.w[query_id]

        query_vec = query_vec.unsqueeze(0)

        similarity = F.cosine_similarity(self.w , query_vec)

        result = similarity.argsort()

        for i in range(top):
            print(idx2word[int(result[i])] , similarity[int(result[i])])

class SGD:
    '''
    (Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]