import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear
from model.w2v_np import HS_skipgram, BCELossWithSigmoid
from preprocess import *
from optim.optimizer import SGD
from itertools import repeat
import time
import pickle

batch_size = 1
sample_size = 5
path = "./1-billion-word/training-monolingual.tokenized.shuffled"

'''
word2idx, idx2word, count, file_path = recall_and_corpus(path)
save = {"word2idx" : word2idx, "idx2word" : idx2word, "count" : count , "file_path" : file_path}

with open("./data.pickle",'wb') as f:
    pickle.dump(save, f)
'''

with open("./data.pickle", 'rb') as f:
    save = pickle.load(f)

word2idx = save["word2idx"]
idx2word = save["idx2word"]
count = save["count"]
file_path = save["file_path"]




node, max_depth = Huffman_Tree(count)


def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words += word

    return words

#sample_size random하게 바꿈. --> 한 corpus에서 더 다양하게 보는게 가능!
model = HS_skipgram(len(word2idx), 300, lr = 0.0025)
criterion = BCELossWithSigmoid()
optimizer = SGD(lr = 0.0025)



'''
words = recall_word(path)
word2idx, idx2word, count = corpus_making(words, most_common= 80000)
word_id = word_id_gen(words, word2idx, count)
node, max_depth = Huffman_Tree(count)
'''


dev = 33
best_loss = 1e10
st = time.time()
for i in range(len(file_path) // dev):
    words = batch_words(file_path[i * dev : (i+1) * dev])
    total_num = len(words)
    word_id = word_id_gen(words, word2idx, count)

    #idx = np.random.permutation(total_num - sample_size)
    cur_t = time.time()
    for iteration in range(total_num - sample_size):
        #print("init",time.time() - st)

        #idx = np.random.choice(total_num - sample_size, batch_size)

        train_data, label = train_token_gen(word_id, sample_size, iteration)
        label = node[label]
        #label = batch_size x (2 * max_depth + 1)
        #print("random",time.time() - st)
        l = 0
        for lab in label:
            y, target = model.forward([train_data], lab)
            loss = criterion.forward(y, target)
            #print("forward", time.time() - st)
            dloss = criterion.backward()
            model.backward(dloss)
            #print("back", time.time() - st)
            l += loss


        #optimizer.update(model.params, model.grads)
        #optimizer._zero_grad(model.grads)

        loss = l / batch_size / 2 / sample_size
        if iteration % 100000 == 0:
            print(f"iteration : {iteration} | current loss : {loss} | time spend : {time.time() - st} | time_left : {total_num / (1+iteration) *(time.time() - cur_t) / 3600} hours left")

            if best_loss > loss:
                best_loss = loss
                model.save("./bestmodel.pickle")
            #ex = np.random.choice(len(word2idx), 1)[0]
            #model.query(idx2word[ex], word2idx, idx2word, top = 5)


def evaluate(path, model, word2idx, idx2word):
    with open("./bestmodel.pickle", 'rb') as f:
        x = pickle.load(f)


    model.params = x
    for i in range(20,30):
        #ex = np.random.choice(len(word2idx), 1)[0]
        model.query(idx2word[i], word2idx, idx2word, top = 15)



#evaluate("",model,word2idx, idx2word)