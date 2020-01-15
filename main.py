import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from utils import _Huffman_Tree, corpus_making, delete_low_freq, train_data_gen
from model.word2vec import CBOW, skip_gram, skip_gram_with_Hierarchy
import time
import tqdm
import torch.nn.functional as F

def data_gen(path, load, sample_size):

    word2idx , idx2word = corpus_making(path, load= False, batch= 50000, save= False)

    word2idx, idx2word = delete_low_freq(word2idx, idx2word, 6)

    train_set_idx, total_word_len = train_data_gen(path = path, word2idx = word2idx, max_distance= sample_size, load = False, batch = 50000, save = False)

    
    return word2idx, idx2word, train_set_idx

def train(model, device, criterion, optimizer, epochs, batch_size, train_mode, sample_size, train_set_idx):

    total_num = len(train_set_idx)

    model.to(device)
    model.train()

    best_loss = 1e10

    for epoch in range(epochs + 1):
        epoch_loss = 0
        loss = 0

        for iteration in range(total_num // batch_size):
            seed = np.random.choice(total_num, batch_size)
            batch_train = train_set_idx[seed]
            #print(batch_train)

            if train_mode.lower() == "cbow":
                y_train = batch_train[:,sample_size]
                temp = [x for x in range(2*sample_size + 1)]
                temp.pop(sample_size)
                x_train = batch_train[:,temp]

                y_train = torch.Tensor(y_train).to(torch.long).to(device)
                x_train = torch.Tensor(x_train).to(torch.long).to(device)

                y_pred = model(x_train)
                optimizer.zero_grad()

                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

            elif train_mode.lower() == "skip":

                x_train = batch_train[:, sample_size]

                y_train = np.delete(batch_train , sample_size, axis =1)

                x_train = torch.Tensor(x_train).to(torch.long).to(device)
                y_train = torch.Tensor(y_train).to(torch.long).to(device)

                loss_batch = 0
                for i in range(2*sample_size):
                    y_pred = model(x_train)

                    #나중에 unigram 으로 sampling 해야함
                    y_target = y_train[:,np.random.choice(2*sample_size, 1)]
                    #y_target squeeze [N, 1] --> [N,]
                    y_target = y_target.squeeze(1)
                    optimizer.zero_grad()
                    #print(y_pred.shape, y_target.shape)
                    loss = criterion(y_pred , y_target)
                    loss.backward()
                    optimizer.step()

                    loss_batch += loss.item()
                loss_batch /= 2*sample_size

            print(f"total iteration : {total_num // batch_size} | iteration : {iteration} | train_iter_loss : {loss}")
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
    train_mode = "cbow"
    C = 5

    word2idx, idx2word, train_set_idx = data_gen(path, load = False, sample_size= C)

    vocab_size = len(word2idx)

    print(f"Total Train Data number : {len(train_set_idx)} | testing the Train_data : train_set_idx = {train_set_idx[np.random.choice(len(train_set_idx), 3)]}")

    #model = CBOW(vocab_size = vocab_size, projection_layer= embed_size, sample_size = C)
    model = skip_gram_with_Hierarchy(vocab_size = vocab_size, projection_layer = embed_size, sample_size = C, device = device)
    #optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-8)

    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    #criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()


    #train(model, device, criterion, optimizer, epochs, batch_size, train_mode, C, train_set_idx)
    #evaluate(model, batch_size, 5, word2idx, idx2word)
    heirarchical_train(model, device, criterion, optimizer, epochs, batch_size, C, train_set_idx, word2idx)

def heirarchical_train(model, device, criterion, optimizer, epochs, batch_size, sample_size, train_set_idx, word2idx):

    b_tree, max_ = _Huffman_Tree(word2idx)

    #x_inputs = np.array([[idx, path] for _, idx, _, path in b_tree])
    label = np.array([[path, idx_path] for _, _, idx_path ,path in b_tree])
    
    total_num = len(train_set_idx)

    model.to(device)
    model.train()

    best_loss = 1e10

    for epoch in range(epochs + 1):
        epoch_loss = 0
        loss = 0

        for iteration in range(total_num // batch_size):
            #batch_train : [N, 2*C + 1]
            batch_train = train_set_idx[np.random.choice(total_num, batch_size)]

            x_train = batch_train[:, sample_size]
            label_train_idx = np.delete(batch_train , sample_size, axis = 1)

            #N x [path(2C), idx_path(2C)]
            label_train = [label[idx] for idx in label_train_idx]

            #x_train : N x [idx, paths]
            #label_train : N x [idx_paths]
            output, target = model(x_train, label_train)
            
            loss = 0
            for out, tar in zip(output, target):
                for o,t in zip(out,tar):
                    loss = loss + criterion(o,t)
        
            loss = loss / (batch_size * 2 * sample_size)

            if iteration % 20000 == 0:
                print(f"total iteration : {total_num // batch_size} | iteration : {iteration} | train_iter_loss : {loss}")
            epoch_loss += loss.item()

        epoch_loss /= (total_num // batch_size)

        print(f"epoch : {epoch} | train_loss : {epoch_loss}")

        #Save model
        best_model = (epoch_loss < best_loss)
        if best_model:
            best_loss = epoch_loss
            torch.save(model, "./best_model.pt")



if __name__ == "__main__":
    main()