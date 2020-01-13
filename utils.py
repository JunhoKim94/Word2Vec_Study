import numpy as np
from tqdm import tqdm
import pickle
import heapq

def corpus_making(path, batch = 500000, save = True, load = False):
    '''
    word2idx : key = word values = [idx, freq]
    idx2word : key = idx values = [word,freq]
    total_word_len : 중복을 허락한 모든 단어의 개수 (Unigram 계산을 위해 return)
    '''

    if load:
        with open("./trunk/word2idx.pickle", 'rb') as f:
            word2idx = pickle.load(f)
        with open("./trunk/idx2word.pickle", 'rb') as f:
            idx2word = pickle.load(f)

        
    else:
        idx2word = dict()
        word2idx = dict()
        batch = batch
        with open(path,'r') as p:
            for line in p:
                total_len = len(line)
        
        print(total_len)

        total_word_len = 0
        with open(path, 'r') as p:
            #while p.readline(1) != "":
            for i in tqdm(range(total_len//batch), desc = "MAKING CORPUS"):
                
                temp = p.readline(batch)
                if temp == "":
                    break
                temp = temp.split(" ")
                total_word_len += len(temp)
                for word in temp:
                    #print(word)
                    if word2idx.get(word) is None:
                        word2idx[word] = [len(word2idx), 1]
                        idx2word[len(idx2word)] = [word, 1]
                    else:
                        #print(word2idx, idx2word)
                        word2idx[word][1] += 1
                        idx2word[word2idx[word][0]][1] += 1
    
    #word_freq = sorted(word2idx.values(), key = lambda items : items[1], reverse= False)


    if save:
        with open("./trunk/word2idx.pickle", 'wb') as f:
            pickle.dump(word2idx, f, protocol = pickle.HIGHEST_PROTOCOL)
        with open("./trunk/idx2word.pickle", 'wb') as f:
            pickle.dump(idx2word, f, protocol = pickle.HIGHEST_PROTOCOL)

    return word2idx, idx2word, total_word_len


def train_data_gen(path, max_distance, word2idx, batch = 500000, save = True, load = False):

    word2idx = word2idx
    train_set_idx = []
    batch = batch


    if load:

        with open("./trunk/train_set_idx.pickle", 'rb') as f:
            train_set_idx = pickle.load(f)


    else:

        with open(path,'r') as p:
            for line in p:
                total_len = len(line)

        with open(path, 'r') as p:
            #while p.readline(1) != "":
            for i in tqdm(range(total_len//batch), desc = "Making Train_set"):
                temp = p.readline(batch)
                if temp == "":
                    break
                temp = temp.split(" ")
                train_set = []
                #change words to index
                for word in temp:
                    train_set += [word2idx[word][0]]

                for i, idx in enumerate(range(max_distance, len(temp) - max_distance + 1)):
                    #train_set_idx : [R-C : R : R+C] X N
                    train_set_idx_temp = train_set[i:idx + max_distance + 1]
                    #train_set_idx 마지막 부분이 짤리는 것을 대비하여 0 으로 패딩
                    if (len(train_set_idx_temp) != (2*max_distance + 1)):
                        for i in range(2*max_distance + 1 - len(train_set_idx_temp)):
                            #print(len(word2idx))
                            train_set_idx_temp += [len(word2idx)]
                    train_set_idx.append(train_set_idx_temp)

    if save:
        with open("./trunk/train_set_idx.pickle", 'wb') as f:
            pickle.dump(train_set_idx, f, protocol = pickle.HIGHEST_PROTOCOL)

    return np.array(train_set_idx)



def _Huffman_Tree(word2idx):
    vocab_size = len(word2idx)
    
    heap = sorted(word2idx.values(), key = lambda items : items[1], reverse= False)
    heapq.heapify(heap)
    for i in tqdm(range(vocab_size - 1), desc = "Huffman_Tree"):
        min1 = heapq.heappop(heap)
        min2 = heapq.heappop(heap)
        
        heapq.heappush(heap,[i + vocab_size, min1[1] + min2[1], min1, min2])
        #heap = sorted(heap, key = lambda items : items[1], reverse= False)
        #print(heap)
    
    #heap have nodes
    #node = [idx, freq, left child, right child]
    node_stack = []
    stack = [[heap[0], [], []]]
    max_depth = 0 
    while len(stack) != 0:
        node, direction_path, node_path = stack.pop()
        #print(stack)
        #print(node,direction_path, node_path)

        if node[0] < vocab_size + 1:
            node.append(direction_path)
            node.append(node_path)
            max_depth = max(len(direction_path), max_depth)
            node_stack.append(node)
        else:
            #node_path는 자기 자신 제외 : leaf 노드는 v가 필요 없으니까
            node_num = node[0] - vocab_size
            node_path += [node_num]
            stack.append([node[2], direction_path + [0], node_path])
            stack.append([node[3], direction_path + [1], node_path])

    #node_stack
    #[idx, freq, direction path(list), node_path(list)]
    #path들은 서로 길이가 다르기 때문에 추가적으로 padding을 해줘야 할거 같음
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
        seed = np.random.choice(len(train_idx), sample_size, p = word_p)

        return word_idx[seed]
'''
class HuffmanNode:
    def __init__(self):
        self.node = None
        self.path_by_direction = None
        self.path_by_node = None
        self.node_freq = None

class HuffmanTree:
    def __init__(self, word2idx, idx2word):
        self.node = HuffmanNode()
        self.word2idx = word2idx
        self.idx2word = idx2word
'''

'''
def train_data_gen(path, max_distance, word2idx, batch = 500000, save = True, load = False):

    word2idx = word2idx
    train_set_words = []
    train_set_idx = []
    batch = batch


    if load:

        with open("./trunk/train_set_words.pickle", 'rb') as f:
            train_set_words = pickle.load(f)
        with open("./trunk/train_set_idx.pickle", 'rb') as f:
            train_set_idx = pickle.load(f)


    else:

        with open(path,'r') as p:
            for line in p:
                total_len = len(line)

        with open(path, 'r') as p:
            #while p.readline(1) != "":
            for i in tqdm(range(total_len//batch), desc = "Making Train_set"):
                temp = p.readline(batch)
                if temp == "":
                    break
                temp = temp.split(" ")
                
                for i, idx in enumerate(range(max_distance, len(temp) - max_distance + 1)):
                    #train_set_words : [R-C : R : R+C] X N
                    train_set_words_temp = np.array(temp[i:idx + max_distance + 1])
                    train_set_words.append(train_set_words_temp)
                    train_set_idx_temp = []
                    for word in train_set_words_temp:
                        train_set_idx_temp += [word2idx[word][0]]
                    
                    #train_set_idx 마지막 부분이 짤리는 것을 대비하여 0 으로 패딩
                    if (len(train_set_idx_temp) != (2*max_distance + 1)):
                        for i in range(2*max_distance + 1 - len(train_set_idx_temp)):
                            train_set_idx_temp += [len(word2idx)]
                    train_set_idx.append(train_set_idx_temp)

    if save:
        with open("./trunk/train_set_words.pickle", 'wb') as f:
            pickle.dump(train_set_words, f, protocol = pickle.HIGHEST_PROTOCOL)
        with open("./trunk/train_set_idx.pickle", 'wb') as f:
            pickle.dump(train_set_idx, f, protocol = pickle.HIGHEST_PROTOCOL)

    return train_set_idx
'''