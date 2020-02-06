import numpy as np
from model.layers import *
from model.HS_model import HS_skipgram
from preprocess import *
from optim.optimizer import SGD
from itertools import repeat
import time
import pickle

batch_size = 300
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

#sample_size random하게 바꿈. --> 한 corpus에서 더 다양하게 보는게 가능!
model = HS_skipgram(len(word2idx), 300, lr = 0.0025)
criterion = BCELossWithSigmoid()
#optimizer = SGD(lr = 0.0025)
nsampler = Sampler(count, 0.75)

'''
with open("./bestmodel.pickle", 'rb') as f:
    x = pickle.load(f)

model.params = x
'''

dev = 1
st = time.time()
for i in range(1,len(file_path) // dev):
    words = batch_words(file_path[i * dev : (i+1) * dev])
    total_num = len(words)
    word_id = word_id_gen(words, word2idx, count)

    iteration = 0
    cur_t = time.time()
    for line in word_id:
        iteration += 1
        train_words = nsampler.sub_sampling(line)
    
        train_data = train_token_gen(word_id, sample_size)
        label = train_data[1,:]
        label = node[label]
        #label = batch_size x (2 * max_depth + 1)

        l = 0
        for train,lab in zip(train_data,label):
            print(train, lab)
            truth = lab[-1]
            target = np.expand_dims(lab[:truth],1)
            idx_path = lab[max_depth: max_depth + truth]
            
            y = model.forward(train, idx_path)

            #if (y > MAX_EXP or y < -MAX_EXP):
            #    continue

            loss = criterion.forward(y, target)
            dloss = criterion.backward()
            model.backward(dloss)
            l += loss

        #optimizer.update(model.params, model.grads)
        #optimizer._zero_grad(model.grads)

        loss = l / batch_size / 2 / sample_size
        if iteration % 3000 == 0:
            left = (total_num // batch_size - iteration)
            t = (time.time() - cur_t) / 3600
            predict = left * t / (iteration + 1)
            print(f"iteration : {iteration} | current loss : {loss} | time spend : {time.time() - st} | time_left : {predict} hours left | total expect time : {(left + iteration) * t / (iteration + 1)}")
            model.save("./bestmodel.pickle")


#evaluate("",model,word2idx, idx2word)