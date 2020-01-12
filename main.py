import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from utils import *
from model.word2vec import CBOW, skip_gram
import time
import tqdm


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
batch_size = 500

train_mode = "CBOW"

C = 10

word2idx , idx2word, word_freq = corpus_making(path, load= False, batch= 50000, save= False)
train_set_words, train_set_idx = train_data_gen(path = path, word2idx = word2idx, max_distance= C, load = False, batch = 50000, save = False)

total_num = len(train_set_idx)
vocab_size = len(word2idx)
embed_size = 600

train_set_idx = np.array(train_set_idx)

print(f"testing the Train_data : train_set_idx = {train_set_idx[np.random.choice(total_num, 3)]}")


model = CBOW(vocab_size = vocab_size, projection_layer= embed_size, max_len = C)
#model = skip_gram(vocab_size= vocab_size, projection_layer= embed_size, max_len= C)
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-8)
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

model.to(device)
model.train()

best_loss = 1e10
for epoch in range(epochs + 1):
    epoch_loss = 0

    for iteration in range(total_num // batch_size):
        seed = np.random.choice(total_num, batch_size)
        batch_train = train_set_idx[seed].tolist()
        batch_train = np.array(batch_train)
        #print(batch_train)

        if train_mode.lower() == "cbow":
            y_train = batch_train[:,C]
            temp = [x for x in range(2*C + 1)]
            temp.pop(C)
            x_train = batch_train[:,temp]

            y_train = torch.Tensor(y_train).to(torch.long).to(device)
            x_train = torch.Tensor(x_train).to(torch.long).to(device)

        else:
            x_train = batch_train.pop(C)
            y_train = batch_train


        y_pred = model(x_train)
        optimizer.zero_grad()

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f"total iteration : {total_num // batch_size} | iteration : {iteration} | train_iter_loss : {loss}")
        epoch_loss += loss.item()

    epoch_loss /= batch_size

    print(f"epoch : {epoch} | train_loss : {epoch_loss}")

        #Save model
    best_model = (epoch_loss < best_loss)
    if best_model:
        best_loss = epoch_loss
        torch.save(model, "./best_model.pt")