import torch
import numpy as np
from torch import optim
from preprocess import *
from model.NS_torch import Negative_Sampling
import time
from NS_train import train



print("\n ========================================================> Training Start <==============================================================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
if torch.cuda.device_count() >= 1:
    print(f"\n ====> Training Start with GPU Number : {torch.cuda.device_count()} GPU Name: {torch.cuda.get_device_name(device=None)}")
else:
    print(f"\n ====> Training Start with CPU ")



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

    model = Negative_Sampling(vocab_size = vocab_size, projection= embed_size)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    nsampler = Sampler(count, power = 0.75, k = 5, skip = 5)

    dev = 1

    for i in range(len(file_path) // dev):
        words = batch_words(file_path[i * dev : (i+1) * dev])
        total_num = len(words)
        word_id = word_id_gen(words, word2idx, count)
        train(model, criterion, optimizer, word_id, nsampler)

if __name__ == "__main__":
    main()