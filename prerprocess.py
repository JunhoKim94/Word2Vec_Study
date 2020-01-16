import numpy as np
from tqdm import tqdm
import pickle
import heapq
import collections

def recall_word(path):
    with open(path, 'r') as f:
        words = f.readlines()
        words = words[0].split()

    return words

def corpus_making_version2(words, most_common = 50000):
    
    word2idx = dict()
    idx2word = dict()

    collect = collections.Counter(words)
    print("Words Count complete")
    selected = collect.most_common(most_common - 1)
    count = [["UNK", -1]]
    count.extend(selected)

    data = []
    for word, freq in count:
        data.append([len(word2idx), freq])
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word
        
    count = data

    return word2idx, idx2word, count

def word_id_gen(words, word2idx, count):

    word_id = []
    for word in tqdm(words, desc = "Changing Word to Index"):

        if word not in word2idx:
            word_id += [word2idx["UNK"]]
            count[0][1] += 1
        else:
            word_id += [word2idx[word]]

    word_id = np.array(word_id)

    return word_id

def train_token_gen(word_id, skip_gram, word_index):
    
    #word_index(중심단어)를 제외한 배열 생성
    seed = []
    for i in range(word_index - skip_gram , word_index + skip_gram + 1):
        seed.append(i)
    seed.pop(skip_gram)
    
    train_data = word_id[word_index]
    label = word_id[seed]
    
    return train_data, label

def Huffman_Tree(count):
    vocab_size = len(count)
    
    heap = [[freq, i] for i,freq in count]
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
    return np.array(node_stack), max_depth



if __name__ == "__main__":

    path = "./data/text8.txt"
    words = recall_word(path)
    word2idx, idx2word, count = corpus_making_version2(words)
    word_id = word_id_gen(words, word2idx, count)
    train_data, label = train_token_gen(word_id, 3, 7)
    node, max_depth = Huffman_Tree(count)
    print(count[0])
    labels = node[label]