import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear
from model.w2v_np import Hsoftmax, BCELoss
from utils import _Huffman_Tree, corpus_making, delete_low_freq, train_data_gen
from optim.optimizer import SGD
from itertools import repeat
import time

batch_size = 300
sample_size = 5
path = "./data/text8.txt"
word2idx , idx2word = corpus_making(path,batch= 50000)

word2idx, idx2word = delete_low_freq(word2idx, idx2word, 10)

train_set_idx, total_word_len = train_data_gen(path = path, word2idx = word2idx, max_distance= sample_size, batch = 50000)


total_num = len(word2idx)

model = Hsoftmax(total_num, 400, sample_size)
criterion = BCELoss()
optimizer = SGD(lr = 0.05)

b_tree, max_ = _Huffman_Tree(word2idx)
label = np.array([[path, idx_path] for _, _, idx_path ,path in b_tree])

st = time.time()

for iteration in range(len(train_set_idx) // batch_size):
    batch_train = train_set_idx[np.random.choice(len(train_set_idx), batch_size)]

    
    l = 0
    for i in range(batch_size):
        x_train = batch_train[i,sample_size]
        label_train_idx = np.delete(batch_train[i,:] , sample_size)
        #1 x [path(2C), idx_path(2C)]
        label_train = label[np.random.choice(label_train_idx, sample_size * 2)]

        for x, lab in zip(repeat(x_train), label_train):
            x = [x]
            y, t = model.forward(x, lab)
            loss = criterion.forward(y, t, dim = 0)
            
            dloss = criterion.backward()
            model.backward(dloss)

            l += loss

    optimizer.update(model.params, model.grads)
    optimizer._zero_grad(model.grads)

    if iteration % 500 == 0:
        print(f"iteration : {iteration} | current loss : {l / batch_size / 2 / sample_size} | time spend : {time.time() - st}")