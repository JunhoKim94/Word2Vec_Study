import numpy as np
from preprocess import cosine_similarity
from tqdm import tqdm

def test_words(path = "./data/questions-words.txt"):
    with open(path, 'r', encoding = "UTF8") as f:
        temp = f.readlines()

    semantic_words = []
    syntatic_words = []
    for e in temp:
        t = e[:-1].split(" ")
        if t[1] == "gram1-adjective-to-adverb":
            break
        if t[0] == ":":
            continue
        semantic_words.append(t)
    for e in temp[::-1]:
        t = e[:-1].split(" ")
        if t[1] == "gram1-adjective-to-adverb":
            break
        if t[0] == ":":
            continue
        syntatic_words.append(t)

    return np.array(semantic_words), np.array(syntatic_words)

def cal_score(semantic_words, syntatic_words, model, word2idx, idx2word):
    
    def statement(query):
        for word in query:
            if word not in word2idx:
                return True
        return False
    '''
    sem = semantic_words[np.random.choice(len(semantic_words), sample)]
    syn = syntatic_words[np.random.choice(len(syntatic_words), sample)]
    '''
    sem = semantic_words
    syn = syntatic_words

    W_in , _ = model.params
    score = [0,0]
    batch = 200
    W_in /= np.linalg.norm(W_in, axis = 1, keepdims = True)
    for i,sort in enumerate([syn, sem]):

        j = 0
        for query in tqdm(sort, desc = "scoring"):
            if statement(query):
                continue
            # 1 - 0 + 2 = 3
            query_vec = W_in[word2idx[query[1]]] - W_in[word2idx[query[0]]] + W_in[word2idx[query[2]]]
            query_vec = np.expand_dims(query_vec, 0)
            #query_vec = np.linalg.norm(query_vec, axis = 1, keepdims = True)
            #test[j,:] = query_vec
            target = word2idx[query[3]]
            j += 1
            #오름차순에 의해 정렬
            similarity = np.matmul(query_vec, W_in.T) #np.sum(query_vec * W_in, axis = 1)  #cosine_similarity(test, W_in)
            result = similarity.argsort()[:,-5:]
            if target in result:
                print(1, target , result)
                score[i] += 1

    return score
        

def evaluate(model, word2idx, idx2word):
    rand = np.random.choice(len(word2idx)//500, 20)
    for i in rand:
        model.query(idx2word[i], word2idx, idx2word, top = 5)