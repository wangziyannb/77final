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
        model = Word2Vec(material, vector_size=100, window=3, min_count=2, epochs=15)
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
    criterion = nn.BCELoss()
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
            hs_valid = None
            # total loss
            t_loss = 0
            v_loss = 0
            for x, y in get_batches(train_set, 4):
                # x is input, y is expected output
                # do not accumulate grad between different batches
                opt.zero_grad()
                x = x.unsqueeze(0)
                # y = y[2]
                # input to model and get output
                out, hs = model(x, hs)
                # strip out h.data and list to tuple
                hs = tuple([h.data for h in hs])
                # calculate loss
                # input: (minibatch, c)
                # target: (minibatch, 1)
                # loss = criterion(out[2], y)
                loss = criterion(out, y)
                # back propagation, calculate grads
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                # update weights according to grads
                opt.step()
                # get the float type of loss
                t_loss += loss.item()
                t.set_postfix(loss='{:.6f}'.format(loss.item()))
                t.update(len(x / 100))
            for val_x, val_y in get_batches(valid_set, 4):
                # close dropout layers, batchNorm layers for eval
                model.eval()
                # the same with training part
                val_x = val_x.unsqueeze(0)
                # val_y = val_y.unsqueeze(0)
                # no need for hidden states output
                preds, hs_valid = model(val_x, hs_valid)
                v_loss += criterion(preds, val_y).item()
            valid_loss.append(np.mean(v_loss))
            # open closed layers for continue learning
            model.train()
            train_loss.append(np.mean(t_loss))

            if e % print_every == 0:
                print(f'Epoch {e}:\nTraining Loss: {train_loss[-1]}')
                print(f'Validation Loss: {valid_loss[-1]}')

    plt.figure(figsize=[8., 6.])
    print(train_loss)
    print(valid_loss)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.show()


def buildNP(model, data_set):
    data_set = flatten(data_set)
    result_data_set = []
    result_data_set_vector = []
    for i in range(len(data_set)):
        if model.wv.has_index_for(data_set[i]):
            result_data_set_vector.append(model.wv.get_vector(data_set[i]))
            words = model.wv.similar_by_word(data_set[i], topn=10)
            ont_hot = []
            for j in range(len(model.wv.vectors)):
                ont_hot.append(0)
            ont_hot[model.wv.get_index(data_set[i])] = 1
            for j in range(len(words)):
                w, p = words[j]
                ont_hot[model.wv.get_index(w)] = 1
            result_data_set.append(ont_hot)
    data_set = np.array(result_data_set, dtype=np.float32)
    data_set_vector = np.array(result_data_set_vector, dtype=np.float32)
    return data_set, data_set_vector


def init(material, model):
    if os.path.exists('a.npy') and os.path.exists('b.npy') and os.path.exists('c.npy') and os.path.exists('d.npy'):
        train_set = np.load('a.npy')
        valid_set = np.load('b.npy')
        train_set_vector = np.load('c.npy')
        valid_set_vector = np.load('d.npy')
    else:
        train_set = material[0:round(len(material) * 0.9)]
        valid_set = material[round(len(material) * 0.9):len(material)]
        train_set, train_set_vector = buildNP(model, train_set)
        valid_set, valid_set_vector = buildNP(model, valid_set)
        np.save('a.npy', train_set)
        np.save('b.npy', valid_set)
        np.save('c.npy', train_set_vector)
        np.save('d.npy', valid_set_vector)
    train_data = torch.tensor(train_set, device=torch.device('cuda'), dtype=torch.float32)
    # train_data = F.one_hot(train_data.long(), num_classes=len(words_emb_model.wv.vectors))
    valid_data = torch.tensor(valid_set, device=torch.device('cuda'), dtype=torch.float32)
    # valid_data = F.one_hot(valid_data.long(), num_classes=len(words_emb_model.wv.vectors))
    train_data_vector = torch.tensor(train_set_vector, device=torch.device('cuda'), dtype=torch.float32)
    valid_data_vector = torch.tensor(valid_set_vector, device=torch.device('cuda'), dtype=torch.float32)
    return (train_data, train_data_vector), (valid_data, valid_data_vector)


def test(model, word_emb_model):
    model.cuda(0)
    model.eval()
    first_three = ['the', 'old', 'lady']
    for i in range(len(first_three)):
        first_three[i] = word_emb_model.wv.get_vector(first_three[i])
    first_three = np.array(first_three)
    first_three = torch.tensor(first_three, device='cuda', dtype=torch.float32)
    out, hs = model(first_three, None)
    hs = tuple([h.data for h in hs])
    out = torch.Tensor.detach(torch.Tensor.cpu(out)).numpy()
    result = []
    for i in range(len(out)):
        key = get_pred(out[i], word_emb_model, 1)
        result.append(key)
    for i in range(10):
        input = word_emb_model.wv.get_vector(result[-1])
        input = torch.tensor(input, device='cuda', dtype=torch.float32).unsqueeze(0)
        out, hs = model(input, hs)
        hs = tuple([h.data for h in hs])
        out = torch.Tensor.detach(torch.Tensor.cpu(out)).numpy()
        key = get_pred(out[0], word_emb_model, 1)
        result.append(key)
    # print(word_emb_model.wv.get_index())
    return result


def get_pred(pred, word_emb_model, threshold=1, number=1):
    if threshold == 1:
        # res is a ndarray due to possible multiple max
        res = np.where(pred == np.max(pred))[0]
    else:
        res = np.where(pred > threshold)
        print(res)
        # todo: randomly select one
    key = word_emb_model.wv.index_to_key[int(res[0])]
    return key


if __name__ == '__main__':
    material = load_material('74-0.txt')
    words_emb_model = load_word_embedded('model.bin', material)
    # print(test(torch.load("b.pkl"), words_emb_model))
    train_tensors, valid_tensors = init(material, words_emb_model)
    input_size = 100
    hidden_size = 300
    num_layers = 2
    output_size = len(words_emb_model.wv.vectors)
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.cuda(0)
    train(model, 10, train_tensors, valid_tensors, lr=0.0001)
    torch.save(model, "b.pkl")
