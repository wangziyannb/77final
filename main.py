# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 23:07
# @Author  : Wang Ziyan Yijing Liao
# @File    : main.py
# @Software: PyCharm

from model import LSTM
from gensim.models import Word2Vec
import os
import numpy as np
import torch
from torch import nn, optim
from utils import preprocessing, flatten
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F


def load_material(textpath):
    ls = []
    with open(textpath, 'r', encoding="utf-8") as o:
        for i in o.readlines():
            l = preprocessing(i)
            if len(l) == 0:
                continue
            ls.append(l)
    return ls


def load_word_embedded(filepath, material):
    if os.path.exists(filepath):
        model = Word2Vec.load(filepath)
    else:
        model = Word2Vec(material, min_count=1)
        # print(model)
        # words = model.wv.index_to_key
        # print(model.wv['the'])
        model.save(filepath)
    return model


def get_batches(data, window):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window).
    """
    data, data_vector = data
    L = len(data)
    # for i in range(L - window):
    #     sequence = data_vector[i:i + window - 1]
    #     sequence = sequence.reshape((window - 1) * 100)
    #     forth = data[i + window]
    #     yield sequence, forth
    for i in range(L - window):
        sequence = data_vector[i:i + window - 1]
        forth = data[i + 1:i + window]
        yield sequence, forth


def train(model, epochs, train_set, valid_set, lr=0.001, print_every=1):
    criterion = nn.CrossEntropyLoss()
    # optimizer is Adam
    # optimizer parameters are weights and biases
    opt = optim.Adam(model.parameters(), lr=lr)
    # record for loss
    train_loss = []
    valid_loss = []

    for e in range(epochs):
        with tqdm(total=len(train_set[0])) as t:
            t.set_description('epoch: {}/{}'.format(e, epochs - 1))
            # No need for hidden state for the first time
            hs = None
            # total loss
            t_loss = 0
            v_loss = 0
            for x, y in get_batches(train_set, 4):
                # x is input, y is expected output
                # do not accumulate grad between different batches
                opt.zero_grad()
                x = x.unsqueeze(0)
                # y = y.unsqueeze(0)
                # input to model and get output
                out, hs = model(x, hs)
                # strip out h.data and list to tuple
                hs = tuple([h.data for h in hs])
                # calculate loss
                # input: (minibatch, c)
                # target: (minibatch, 1)
                loss = criterion(out, y)
                # back propagation, calculate grads
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                # update weights according to grads
                opt.step()
                # get the float type of loss
                t_loss += loss.item()
                t.set_postfix(loss='{:.6f}'.format(t_loss))
                t.update(len(x / 100))
            for val_x, val_y in get_batches(valid_set, 4):
                # close dropout layers, batchNorm layers for eval
                model.eval()
                # the same with training part
                val_x = val_x.unsqueeze(0)
                # val_y = val_y.unsqueeze(0)
                # no need for hidden states output
                preds, _ = model(val_x, hs)
                v_loss += criterion(preds, val_y).item()
                valid_loss.append(v_loss)

            # open closed layers for continue learning
            model.train()

            train_loss.append(np.mean(t_loss))

            if e % print_every == 0:
                print(f'Epoch {e}:\nTraining Loss: {train_loss[-1]}')
                print(f'Validation Loss: {valid_loss[-1]}')

    plt.figure(figsize=[8., 6.])
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.show()


def init(material, model):
    train_set = material[0:round(len(material) * 0.8)]
    valid_set = material[round(len(material) * 0.8):len(material)]
    train_set = flatten(train_set)
    valid_set = flatten(valid_set)
    train_set_vector = list(train_set)
    valid_set_vector = list(valid_set)
    for i in range(len(train_set)):
        train_set_vector[i] = model.wv.get_vector(train_set[i])
        train_set[i] = model.wv.get_index(train_set[i])
    for i in range(len(valid_set)):
        valid_set_vector[i] = model.wv.get_vector(valid_set[i])
        valid_set[i] = model.wv.get_index(valid_set[i])
    train_set = np.array(train_set, dtype=np.int32)
    valid_set = np.array(valid_set, dtype=np.int32)
    train_set_vector = np.array(train_set_vector, dtype=np.float32)
    valid_set_vector = np.array(valid_set_vector, dtype=np.float32)

    train_data = torch.tensor(train_set, device=torch.device('cuda'), dtype=torch.int64)
    # train_data = F.one_hot(train_data.long(), num_classes=len(words_emb_model.wv.vectors))
    valid_data = torch.tensor(valid_set, device=torch.device('cuda'), dtype=torch.int64)
    # valid_data = F.one_hot(valid_data.long(), num_classes=len(words_emb_model.wv.vectors))
    train_data_vector = torch.tensor(train_set_vector, device=torch.device('cuda'), dtype=torch.float32)
    valid_data_vector = torch.tensor(valid_set_vector, device=torch.device('cuda'), dtype=torch.float32)

    return (train_data, train_data_vector), (valid_data, valid_data_vector)


if __name__ == '__main__':
    material = load_material('74-0.txt')
    words_emb_model = load_word_embedded('model.bin', material)
    train_tensors, valid_tensors = init(material, words_emb_model)
    input_size = 100
    hidden_size = 300
    num_layers = 2
    output_size = len(words_emb_model.wv.vectors)
    # model = torch.load("a.pkl")
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.cuda(0)

    train(model, 10, train_tensors, valid_tensors, lr=0.00001)
    torch.save(model, "a.pkl")
