import numpy as np
from tqdm import tqdm
import pickle
import heapq
import collections

def corpus_making(path, batch = 500000):
    '''
    word2idx : key = word values = [idx, freq]
    idx2word : key = idx values = [word,freq]
    total_word_len : 중복을 허락한 모든 단어의 개수 (Unigram 계산을 위해 return)
    '''
    idx2word = dict()
    word2idx = dict()
    batch = batch
    with open(path,'r') as p:
        for line in p:
            total_len = len(line)
    
    with open(path, 'r') as p:
        for i in tqdm(range(total_len//batch), desc = "MAKING CORPUS"):
            
            temp = p.readline(batch)
            if temp == "":
                break
            temp = temp.split(" ")

            for word in temp:

                if word2idx.get(word) is None:
                    word2idx[word] = [len(word2idx) + 1, 1]
                    idx2word[len(idx2word)] = [word, 1]

                else:

                    word2idx[word][1] += 1
                    idx2word[word2idx[word][0]][1] += 1
    
    return word2idx, idx2word

def delete_low_freq(word2idx, idx2word, low_freq):

    new_word2idx, new_idx2word = dict(), dict()

    length = len(idx2word)

    for i in range(length):
        temp = idx2word[i]
        if temp[1] < low_freq:
            word2idx.pop(temp[0])
            idx2word.pop(i)

    i = 0
    for key, values in word2idx.items():
        new_word2idx[key] = [i, values[1]]
        new_idx2word[i] = [key, values[1]]
        i += 1

    return new_word2idx, new_idx2word

def train_data_gen(path, max_distance, word2idx, batch = 500000):

    word2idx = word2idx
    train_set_idx = []
    batch = batch

    with open(path,'r') as p:
        for line in p:
            total_len = len(line)

    with open(path, 'r') as p:
        for i in tqdm(range(total_len//batch), desc = "Making Train_set"):
            temp = p.readline(batch)
            if temp == "":
                break
            temp = temp.split(" ")
            train_set = []
            #change words to index
            total_word_len = 0
            for word in temp:
                
                if word not in word2idx.keys():
                    continue

                train_set += [word2idx[word][0]]

            total_word_len += len(train_set)


            for i, idx in enumerate(range(max_distance, len(temp) - max_distance + 1)):
                #train_set_idx : [R-C : R : R+C] X N
                train_set_idx_temp = np.array(train_set[i:idx + max_distance + 1])
                #train_set_idx 마지막 부분이 짤리는 것을 대비하여 0 으로 패딩
                if (len(train_set_idx_temp) != (2*max_distance + 1)):
                    continue
                    '''
                    for i in range(2*max_distance + 1 - len(train_set_idx_temp)):
                        #print(len(word2idx))
                        train_set_idx_temp += [len(word2idx)]
                    '''
                train_set_idx.append(train_set_idx_temp)


    return np.array(train_set_idx), total_word_len



def _Huffman_Tree(word2idx):
    vocab_size = len(word2idx)
    
    heap = [[freq, i] for i,freq in word2idx.values()]
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


    node_stack = sorted(node_stack, key=lambda items: items[1])
    #[idx, freq, direction path(list), node_path(list)]
    return node_stack, max_depth

class Unigram_Sampler:
    def __init__(self, idx2word, power, total_word_len):

        self.vocab_size = len(idx2word)
        self.word_p = np.zeros(self.vocab_size + 1)
        self.power = power

        for i in range(self.vocab_size):
            self.word_p[i + 1] =  idx2word[i + 1][1]

        self.word_p = np.power(self.word_p, self.power)
        self.word_p /=  np.sum(self.word_p)

    def sampling(self, word_idx, sample_size):
        '''
        input :
                word_idx (1 , 2C)
                sample_size = 2C
        output:
                unisampled_word_idx (1 * 2C)
        '''
        word_p = self.word_p[word_idx]
        seed = np.random.choice(len(word_idx), sample_size, p = word_p)

        return word_idx[seed]

if __name__ == "__main__":
    path = "./data/text8.txt"
    word2idx , idx2word = corpus_making(path, batch= 50000)
    word2idx, idx2word = delete_low_freq(word2idx, idx2word, 10)
    n, max_ = _Huffman_Tree(word2idx)

    print(n[0][2] , n[0][3])

