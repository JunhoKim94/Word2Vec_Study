import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CBOW(nn.Module):
    def __init__(self,vocab_size, projection_layer, sample_size):
        super(CBOW, self).__init__()
        '''
        Input : (N,V) one-hot-vectors == words --> embedding
        Output : (1,V) words
        '''
        self.projection_layer = projection_layer
        self.vocab_size = vocab_size
        self.sample_size = sample_size

        self.embedding = nn.Embedding(self.vocab_size + 1, self.projection_layer, padding_idx= self.vocab_size)
        self.linear = nn.Linear(self.projection_layer, self.vocab_size)


    def forward(self, x):
        '''
        x --> (N,C,V) : (N,C) to index
        out --> (N,1,V)
        '''
        
        #x = x.unsqueeze(0)
        #(N,C)
        out = self.embedding(x)
        
        #(N,D)
        out = out.sum(dim = 1)
        
        #(N,V)
        out = self.linear(out)
        
        #out = F.softmax(out, dim = 1)
        #out = F.log_softmax(out, dim = 1)


        return out

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


class skip_gram(nn.Module):
    def __init__(self, vocab_size, projection_layer, sample_size):
        super(skip_gram, self).__init__()

        self.vocab_size = vocab_size
        self.projection_layer = projection_layer
        self.sample_size = sample_size

        #padding data --> idx = 0 : embed size 가 vocab size + 1 이여야 됨
        self.embedding = nn.Embedding(self.vocab_size + 1, self.projection_layer, padding_idx= self.vocab_size)
        self.linear = nn.Linear(self.projection_layer, vocab_size)

    def forward(self, x):
        '''
        x = (N,1,V) : (N,1)
        out = (N,C,V)
        '''
        #(N,D)
        out = self.embedding(x)
        #(N,V)
        out = self.linear(out)
        #out = F.log_softmax(out, dim = 1)
        #out = F.softmax(out, dim = 1)

        return out

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
        
class skip_gram_with_Hierarchy(nn.Module):
    def __init__(self, vocab_size, projection_layer, sample_size):
        super(skip_gram_with_Hierarchy, self).__init__()

        self.vocab_size = vocab_size
        self.projection_layer = projection_layer
        self.sample_size = sample_size

        self.embedding_1 = nn.Embedding(self.vocab_size + 1, self.projection_layer, padding_idx = self.vocab_size)
        
        self.embedding_2 = nn.Embedding(self.vocab_size - 1, self.projection_layer)

    def forward_step(self, x_input):
        '''
        skip-gram 중 random sample 하나 학습
        depth < max_depth
        input : x_input (idx, direction_list), idx_path(depth)
        '''

        x_idx = torch.Tensor([x_input[0]])
        dir_path = torch.Tensor(x_input[1])

        #(1,D) : hidden layer
        proj = self.embedding_1(x_idx)

        #(path_length, D)
        hirearchy_vectors = self.embedding_2(dir_path)

        #(1, path_length)
        output = torch.matmul(proj, hirearchy_vectors.T)
        output = torch.sigmoid(output)

        return output

    def forward(self, inputs, label):
        '''
        inputs : N x [idx, direction_list]
        label : N x idx_path
        label 과 output 의 argmax를 비교해서 같으면 1 틀리면 0 을 부여한 후 이를 target vector로 설정해야됨
        ex) output = [0.7, 0.3, 0.4] label = [1, 1, 0]
        --> target = [1, 0, 1]
        
        : BCE Loss 를 사용할 것 : - y_t * log(y_p) - (1-y_t) * log(1 - y_p)
        false --> -log(1 - y_p) = -log(sigmoid(-v_t * h))
        True --> -log(y_p) = -log(sigmoid(v_t*h))
        '''

        output = self.forward_step(inputs)

        mask = torch.zeros_like(output)
        mask[output >= 0.5] = 1

        target = torch.zeros_like(label)
        target[mask == label] = 1

        return output, target

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
        
