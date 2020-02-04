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

def cal_score(sample , semantic_words, syntatic_words, model, word2idx, idx2word):
    
    def statement(query):
        for word in query:
            if word not in word2idx:
                return True
        return False

    sem = semantic_words[np.random.choice(len(semantic_words), sample)]
    syn = syntatic_words[np.random.choice(len(syntatic_words), sample)]

    W_in , _ = model.params
    score = [0,0]
    
    for i,sort in enumerate([syn, sem]):
        test = np.zeros((sample, 300))

        for j,query in enumerate(sort):
            if statement(query):
                continue
            # 1 - 0 + 2 = 3
            print(query)
            query_vec = W_in[word2idx[query[1]]] - W_in[word2idx[query[0]]] + W_in[word2idx[query[2]]]
            test[j,:] = query_vec
        
        #오름차순에 의해 정렬
        similarity = cosine_similarity(test, W_in)
        result = similarity.argsort(axis = 1)[:,-5:]

        for idx, ans in zip(result, sort[:,3]):
            print("answear : ", ans)
            for j in idx:
                print("candidate : ",idx2word[j])
            if statement([ans]):
                continue
            if word2idx[ans] in idx:
                score[i] += 1
    return score

        

def evaluate(model, word2idx, idx2word):
    rand = np.random.choice(len(word2idx), 20)
    for i in rand:
        model.query(idx2word[i], word2idx, idx2word, top = 5)