import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from preprocess import *
from model.word2vec import CBOW, skip_gram, skip_gram_with_Hierarchy
import time
import tqdm
import torch.nn.functional as F

def data_gen(path, load, sample_size):

    words = recall_word(path)
    word2idx, idx2word, count = corpus_making_version2(words)
    word_id = word_id_gen(words, word2idx, count)

    return word2idx, idx2word, word_id

def train(model, device, criterion, optimizer, epochs, batch_size, sample_size, word_id):

    total_num = len(word_id)

    model.to(device)
    model.train()

    best_loss = 1e10

    for epoch in range(epochs + 1):
        epoch_loss = 0
        loss = 0

        for iteration in range(total_num // batch_size):

            idx = np.random.choice(total_num, batch_size)
            train_data, label = train_token_gen(word_id, sample_size, idx)

            #섞섞
            seed = np.random.permutation(len(train_data))
            
            train_data = train_data[seed]
            label = label[seed]

            y_train = torch.Tensor(label).to(torch.long).to(device)
            x_train = torch.Tensor(train_data).to(torch.long).to(device)

            y_pred = model(x_train.flatten())

            optimizer.zero_grad()

            loss = criterion(y_pred, y_train)
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

def evaluate(model, batch_size, top, word2idx, idx2word):

    model.eval()
    total_num = len(idx2word)
    seed = np.random.choice(total_num, batch_size)
    
    for i in seed:
        word = idx2word[i][0]
        print(word)
        model.query(word, word2idx, idx2word, top = 5)
        print("next batch")       

 
def main():
    print("\n ========================================================> Training Start <==============================================================")
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    print(torch.cuda.is_available())
    if torch.cuda.device_count() >= 1:
        print(f"\n ====> Training Start with GPU Number : {torch.cuda.device_count()} GPU Name: {torch.cuda.get_device_name(device=None)}")
    else:
        print(f"\n ====> Training Start with CPU ")


    np.random.seed(19941017)

    path = "./data/text8.txt"
    learning_rate  = 0.025
    epochs = 3
    embed_size = 640
    batch_size = 1
    C = 5

    word2idx, idx2word, word_id = data_gen(path, load = False, sample_size= C)

    vocab_size = len(word2idx)

    print(f"Total Train Data number : {len(word_id)} | epoch : {epochs} | embed size : {embed_size} | batch size : {batch_size}")

    #model = CBOW(vocab_size = vocab_size, projection_layer= embed_size, sample_size = C)
    #optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-8)

    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    #criterion = torch.nn.BCELoss()

    #train(model, device, criterion, optimizer, epochs, batch_size, train_mode, C, train_set_idx)
    #evaluate(model, batch_size, 5, word2idx, idx2word)

if __name__ == "__main__":
    main()