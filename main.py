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
import random
from sklearn.manifold import TSNE
import pandas as pd


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
        model = Word2Vec(material, vector_size=100, window=5, min_count=2, epochs=15)
        model.save(filepath)
    return model


def show_word_embedded(model):
    vocab = list(model.wv.key_to_index)
    X = model.wv[vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    print(df.head())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    plt.show()


def get_batches(data, window):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window).
    """
    data, data_vector = data
    L = len(data)
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
                # input to model and get output
                out, hs = model(x, hs)
                # strip out h.data and list to tuple
                hs = tuple([h.data for h in hs])
                # calculate loss
                # input: (minibatch, c)
                # target: (minibatch, 1)
                loss = criterion(out[2], y[2])
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
            torch.save(model, "epoch" + str(e) + ".pkl")
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
    # data input vector and label
    result_data_set = []
    result_data_set_vector = []
    for i in range(len(data_set)):
        if model.wv.has_index_for(data_set[i]):
            # append the input vector
            result_data_set_vector.append(model.wv.get_vector(data_set[i]))
            # select top 10 similar word and the ground truth word as the expected label
            words = model.wv.similar_by_word(data_set[i], topn=10)
            one_hot = []
            # vector shape: [1,6900]
            for j in range(len(model.wv.vectors)):
                one_hot.append(0)
            # ground truth
            one_hot[model.wv.get_index(data_set[i])] = 1
            for j in range(len(words)):
                w, p = words[j]
                one_hot[model.wv.get_index(w)] = 1
            result_data_set.append(one_hot)
    data_set = np.array(result_data_set, dtype=np.float32)
    data_set_vector = np.array(result_data_set_vector, dtype=np.float32)
    return data_set, data_set_vector


def init(material, model):
    # if there are processed data, load the data
    if os.path.exists('a.npy') and os.path.exists('b.npy') and os.path.exists('c.npy') and os.path.exists('d.npy'):
        train_set = np.load('a.npy')
        valid_set = np.load('b.npy')
        train_set_vector = np.load('c.npy')
        valid_set_vector = np.load('d.npy')
    else:
        train_set = material[0:round(len(material) * 0.9)]
        valid_set = material[round(len(material) * 0.9):len(material)]
        # build the input vector and the expected classification label
        train_set, train_set_vector = buildNP(model, train_set)
        valid_set, valid_set_vector = buildNP(model, valid_set)
        np.save('a.npy', train_set)
        np.save('b.npy', valid_set)
        np.save('c.npy', train_set_vector)
        np.save('d.npy', valid_set_vector)
    train_data = torch.tensor(train_set, device=torch.device('cuda'), dtype=torch.float32)
    valid_data = torch.tensor(valid_set, device=torch.device('cuda'), dtype=torch.float32)
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
    key = get_pred(out[2], word_emb_model, 0.2)
    result.append(key)
    # uncomment if you want a pred sequence
    # for i in range(10):
    #     input = word_emb_model.wv.get_vector(result[-1])
    #     input = torch.tensor(input, device='cuda', dtype=torch.float32).unsqueeze(0)
    #     out, hs = model(input, hs)
    #     hs = tuple([h.data for h in hs])
    #     out = torch.Tensor.detach(torch.Tensor.cpu(out)).numpy()
    #     key = get_pred(out[0], word_emb_model, 1)
    #     result.append(key)
    # print(word_emb_model.wv.get_index())
    return result


def get_pred(pred, word_emb_model, threshold=1.0, number=2):
    if threshold == 1.0:
        # res is a ndarray due to possible multiple max
        res = np.argmax(pred)
    else:
        res = np.where(pred > threshold)[0]
        if len(res) == 0:
            res = np.argpartition(pred, -number)
            res = res[-number:]
        res = random.choice(res)
    key = word_emb_model.wv.index_to_key[res]

    return key


if __name__ == '__main__':
    # material loading
    material = load_material('74-0.txt')
    # material = material + load_material('55-0.txt')

    # words embedding
    words_emb_model = load_word_embedded('model.bin', material)
    # show_word_embedded(words_emb_model)
    # data tensors preparing
    train_tensors, valid_tensors = init(material, words_emb_model)

    # network configuration&definition
    input_size = 100
    hidden_size = 300
    num_layers = 2
    output_size = len(words_emb_model.wv.vectors)
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.cuda(0)

    # model training
    train(model, 50, train_tensors, valid_tensors, lr=0.00005)

    # word prediction(with trained model)
    print(test(torch.load("epoch49.pkl"), words_emb_model))
