import numpy as np
from tqdm import tqdm
import pickle
import heapq
import collections
import os

class Sampler:
    def __init__(self, count , power, k = 5, skip = 5):

        self.vocab_size = len(count)
        self.count = np.array(count)
        self.power = power

        self.k = k
        self.skip = skip * 2

        self.word_p = self.count / np.sum(self.count)
        self.sub_word_p = 1 - (1e-5 / self.word_p) ** 0.5
        self.sub_word_p[self.sub_word_p < 0] = 0
        #"UNK" token --> always exclude
        self.sub_word_p[0] = 1

        #self.word_p /= np.sum(self.word_p)
        self.word_p = np.power(self.word_p, self.power)
        self.word_p[0] = 0
        self.word_p = self.word_p / np.sum(self.word_p)

    def nega_train_token(self, word_id, word_index):
        #word_index(중심단어)를 제외한 배열 생성
        batch_size = len(word_index)
        
        negative_sample = np.zeros((batch_size, (self.k + 1) * self.skip), dtype = np.int32)
        target = np.zeros((batch_size, (self.k + 1) * self.skip))

        for i, index in enumerate(word_index):

            seed = list(range(index - self.skip//2 , index + self.skip//2 + 1))
            seed.pop(self.skip//2)
            tar_idx = word_id[seed]

            #1 x k *skip_size
            p = self.word_p.copy()
            p[tar_idx] = 0
            p /= p.sum()
            negative_sample[i, :self.skip] = tar_idx
            negative_sample[i, self.skip:] = np.random.choice(self.vocab_size, self.k * self.skip, p = p)
            target[i, :self.skip] = 1

        return negative_sample, target

    def sub_sampling(self, word_idx):
        '''
        word_idx = 1 x batch --> [i1, i2, i3, i1, i1, i2 ...]
        out --> [T, F, F, T, F ...]
        '''
        def subtract(probablity):
            return np.random.choice([False,True] , p = [probablity, 1 - probablity])
        iswordtrain = []
        for idx in word_idx:
            probablity = self.sub_word_p[idx]
            iswordtrain.append(subtract(probablity))

        return iswordtrain

def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words += word

    return words

def recall_word(path):

    word = []
    with open(path, 'r', encoding = "UTF8") as f:
        lines = f.readlines()
        for line in lines:
            word += line.split()

    return word

#대용량 데이터를 나눠서 corpus를 생성
def recall_and_corpus(path, batch = 9):

    file_path = list()
    for path, dir , files in os.walk(path):
        for filename in files:
            file_path.append(path + "/" + filename)


    assert len(file_path) % batch == 0

    collect = collections.Counter()

    del path, dir, files

    for i in tqdm(range(len(file_path)//batch), desc =  "Making Corpus by batch"):
        word_token = []
        for path in file_path[i * batch : (i+1) * batch]:
            word = recall_word(path)
            word_token += word

        collect.update(word_token)
    
    temp = collect.most_common()
    selected = collect.most_common(692 * 10**3)
    count = [["UNK", 1]]
    count.extend(selected)

    word2idx = dict()
    idx2word = dict()
    data = []
    for word, freq in count:
        data.append(freq)
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word
        
    count = data

    return word2idx, idx2word, count, file_path


def corpus_making(words, most_common = 50000):
    
    word2idx = dict()
    idx2word = dict()

    collect = collections.Counter(words)
    print("Words Count complete")
    selected = collect.most_common(most_common - 1)
    count = [["UNK", -1]]
    count.extend(selected)

    data = []
    for word, freq in count:
        data.append(freq)
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word
        
    count = data

    return word2idx, idx2word, count

def word_id_gen(words, word2idx, count):

    word_id = []
    for word in tqdm(words, desc = "Changing Word to Index"):

        if word not in word2idx:
            word_id += [word2idx["UNK"]]
            count[0] += 1
        else:
            word_id += [word2idx[word]]

    word_id = np.array(word_id)

    return word_id

def train_token_gen(word_id, skip_gram, word_index):
    
    #word_index(중심단어)를 제외한 배열 생성
    batch_size = len(word_index)
    skip_size = skip_gram * 2
    
    train_data = np.ndarray(shape = (batch_size * skip_size, 1), dtype = np.int32)
    label = np.ndarray(shape = (batch_size * skip_size), dtype = np.int32)
    
    
    for i, index in enumerate(word_index):

        seed = list(range(index - skip_gram , index + skip_gram + 1))
        seed.pop(skip_gram)

        train_data[i * skip_size : (i+1) * skip_size, 0] = word_id[index]
        label[i * skip_size : (i+1) * skip_size] = word_id[seed]
    
    #seed = list(range(word_index - skip_gram , word_index + skip_gram + 1))
    #seed.pop(skip_gram)
    #return word_id[word_index], word_id[seed]

    return train_data, label

def Huffman_Tree(count):
    vocab_size = len(count)
    
    heap = [[freq, i] for i,freq in enumerate(count)]
    heapq.heapify(heap)
    for i in tqdm(range(vocab_size - 1), desc = "Huffman_Tree"):
        min1 = heapq.heappop(heap)
        min2 = heapq.heappop(heap)
        
        heapq.heappush(heap,[min1[0] + min2[0], i + vocab_size, min1, min2])

    #node = [idx, freq, left child, right child]
    node_stack = []
    stack = [[heap[0], [], []]]
    max_depth = 0 
    while len(stack) != 0:
        node, direction_path, node_path = stack.pop()

        if node[1] < vocab_size:
            node.append(np.array(direction_path))
            node.append(np.array(node_path))
            max_depth = max(len(direction_path), max_depth)
            node_stack.append(node)
        else:
            #node_path는 자기 자신 제외 : leaf 노드는 v가 필요 없으니까
            node_num = [node[1] - vocab_size]
            stack.append([node[2], direction_path + [0], node_path + node_num])
            stack.append([node[3], direction_path + [1], node_path + node_num])


    node_stack = np.array(sorted(node_stack, key=lambda items: items[1]))
    node_stack = node_stack[:, 2:4]

    #path = ([dir_path + pad],[node_path + pad],truth_length)
    path_ = np.zeros((vocab_size, max_depth * 2 + 1)).astype(np.int)
    for i in tqdm(range(vocab_size), desc = "Padding Paths"):
        truth_length = len(node_stack[i,0])
        path_[i,0:truth_length] = node_stack[i,0]
        path_[i, max_depth : max_depth + truth_length] = node_stack[i,1]
        path_[i, -1] = truth_length
    #[direction path(list), node_path(list)]
    return path_, max_depth

def cosine_similarity(x,y):
    # 두 벡터간의 cos (theta) 를 통해 유사도 결정
    assert len(x.shape) == 2
    # y --> (M, D) : M개 검사

    #product = np.sum(x * y , axis = 1)
    
    x /= np.linalg.norm(x , axis = 1, keepdims = True)
    y /= np.linalg.norm(y, axis = 1, keepdims = True)

    product = np.matmul(x, y.T)

    return product
