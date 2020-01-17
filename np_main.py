import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear
from model.w2v_np import Hsoftmax, BCELoss
from preprocess import *
from optim.optimizer import SGD
from itertools import repeat
import time

batch_size = 300
sample_size = 5
path = "./data/text8.txt"

words = recall_word(path)
word2idx, idx2word, count = corpus_making_version2(words, most_common= 80000)
word_id = word_id_gen(words, word2idx, count)

node, max_depth = Huffman_Tree(count)

total_num = len(words)

model = Hsoftmax(len(word2idx), 400, sample_size)
criterion = BCELoss()
optimizer = SGD(lr = 0.05)

st = time.time()



for iteration in range(total_num // batch_size):

    idx = np.random.choice(total_num - sample_size, batch_size)
    train_data, label = train_token_gen(word_id, sample_size, idx)
    label = node[label]

    l = 0
    for x, lab in zip(train_data, label):
        y, t = model.forward(x, lab)
        loss = criterion.forward(y, t, dim = 0)
        
        dloss = criterion.backward()
        model.backward(dloss)

        l += loss

    optimizer.update(model.params, model.grads)
    optimizer._zero_grad(model.grads)

    
    if iteration % 500 == 0:
        print(f"iteration : {iteration} | current loss : {l / batch_size / 2 / sample_size} | time spend : {time.time() - st}")
        model.save("./bestmodel.pickle")
        ex = np.random.choice(len(word2idx), 1)[0]
        model.query(idx2word[ex], word2idx, idx2word, top = 5)



def evaluate(path, model, word2idx, idx2word):
    with open("./bestmodel.pickle", 'rb') as f:
        x = pickle.load(f)

    model.params = x
    for i in range(10):
        ex = np.random.choice(len(word2idx), 1)[0]
        model.query(idx2word[ex], word2idx, idx2word, top = 5)