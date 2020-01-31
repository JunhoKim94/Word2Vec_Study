import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, BCELossWithSigmoid
from model.NS_model import Negative_Sampling
from preprocess import *
import time
import pickle

np.random.seed(19941017)

learning_rate  = 0.025
epochs = 3
embed_size = 300
batch_size = 300
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

def train(model, criterion, word_id, nsampler):

    total_num = len(word_id)
    
    best_loss = 1e10

    for epoch in range(epochs + 1):
        epoch_loss = 0
        loss = 0
        
        curr_time = time.time()
        for iteration in range(total_num // batch_size):

            idx = np.random.choice(total_num - sample_size, batch_size)
            judge = nsampler.sub_sampling(word_id[idx])
            idx = idx[judge]

            #(N,1), (N,S), (N, S)
            nega_samples, target = nsampler.nega_train_token(word_id, idx)
            train_data = word_id[idx]
            #N,S
            y_pred = model.forward(train_data, nega_samples)
            loss = criterion.forward(y_pred, target)
            dloss = criterion.backward()
            model.backward(dloss)

        if iteration % 300 == 0:
            left = (total_num // batch_size - iteration)
            t = (time.time() - cur_t) / 3600
            predict = left * t / (iteration + 1)
            print(f"iteration : {iteration} | current loss : {loss} | time spend : {time.time() - st} | time_left : {predict} hours left | total expect time : {(left + iteration) * t / (iteration + 1)}")
            model.save("./bestmodel.pickle")
 
def train_torch(model, criterion, optimizer, word_id, nsampler):

    total_num = len(word_id)
    

    model.to(device)
    model.train()

    best_loss = 1e10

    for epoch in range(epochs + 1):
        epoch_loss = 0
        loss = 0
        
        curr_time = time.time()
        for iteration in range(total_num // batch_size):

            idx = np.random.choice(total_num - sample_size, batch_size)
            judge = nsampler.sub_sampling(word_id[idx])
            idx = idx[judge]

            #(N,1), (N,S), (N, S)
            print("subsampling time : ", time.time()- curr_time)
            nega_samples, target = nsampler.nega_train_token(word_id, idx)
            train_data = word_id[idx]
            print("nega sampling time : ", time.time() - curr_time)

            train_data = torch.LongTensor(train_data).to(device)
            nega_samples = torch.LongTensor(nega_samples).to(device)
            target = torch.Tensor(target).to(device)

            #N,S
            y_pred = model(train_data, nega_samples)

            optimizer.zero_grad()

            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()


            #print(f"total iteration : {total_num // batch_size} | iteration : {iteration} | train_iter_loss : {loss}")
            epoch_loss += loss.item()

        epoch_loss /= (total_num // batch_size)

        print(f"epoch : {epoch} | train_loss : {epoch_loss}")

        #Save model
        best_model = (epoch_loss < best_loss)
        if best_model:
            best_loss = epoch_loss
            torch.save(model, "./best_model.pt")

def main():

    word2idx, idx2word, count, file_path = data_gen()
    vocab_size = len(word2idx)

    model = Negative_Sampling(vocab_size = vocab_size, projection= embed_size, lr = 0.0025)
    criterion = BCELossWithSigmoid()
    nsampler = Sampler(count, power = 0.75, k = 5, skip = 5)

    dev = 1

    for i in range(len(file_path) // dev):
        words = batch_words(file_path[i * dev : (i+1) * dev])
        total_num = len(words)
        word_id = word_id_gen(words, word2idx, count)
        train(model, criterion, word_id, nsampler)

if __name__ == "__main__":
    main()