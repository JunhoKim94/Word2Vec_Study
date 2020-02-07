import torch
import numpy as np
from torch import optim
from preprocess import *
from model.HS_model import HS_skipgram
import time
from eval import *
from model.layers import BCELossWithSigmoid


np.random.seed(19941017)
learning_rate  = 0.025
epochs = 3
embed_size = 300
batch_size = 500
sample_size = 5
k = 5

def data_gen(path = "./1-billion-word/training-monolingual.tokenized.shuffled", load = True):
    if load:
        with open("./data.pickle", 'rb') as f:
            save = pickle.load(f)

        word2idx = save["word2idx"]
        idx2word = save["idx2word"]
        count = save["count"]
        file_path = save["file_path"]

        return word2idx, idx2word, count, file_path

    word2idx, idx2word, count, file_path = recall_and_corpus(path)
    save = {"word2idx" : word2idx, "idx2word" : idx2word, "count" : count , "file_path" : file_path}

    with open("./data.pickle",'wb') as f:
        pickle.dump(save, f)

    return word2idx, idx2word, count, file_path

def main():

    word2idx, idx2word, count, file_path = data_gen()
    vocab_size = len(word2idx)

    model = HS_skipgram(vocab_size = vocab_size, projection= embed_size, lr = learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    #criterion = torch.nn.BCEWithLogitsLoss()
    #nsampler = Sampler(count, power = 0.75, k = 5, skip = 5)

    with open("./bestmodel.pickle", 'rb') as f:
        x = pickle.load(f)


    model.params = x
    sem, syn = test_words()
    print(sem, syn)
    a = cal_score(sem,syn, model, word2idx, idx2word)
    #evaluate(model, word2idx, idx2word)
    print(a)
if __name__ == "__main__":
    main()


    #a = np.ones(shape = (10,5))
    #print(exp[a])
    #print(exp[a].shape)