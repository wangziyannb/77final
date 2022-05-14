# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 4:13
# @Author  : Wang Ziyan Yijing Liao
# @File    : lstm.py
# @Software: PyCharm
import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hs):
        # input shape: (batch_size, seq_length, num_features), and hs for hidden state
        # out:(batch_size, seq_length, hidden_size), (hn, cn)
        out, hs = self.lstm(x, hs)
        # reshape our data into the form (batches, n_hidden)
        out = out.reshape(-1, self.hidden_size)
        # out = out.view(-1, self.hidden_size)
        # input shape: (batch_size * seq_length, hidden_size)
        out = self.fc(out)
        # output shape: (batch_size * seq_length, out_size)
        return out, hs


def get_batches(data, window):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window).
    """

    L = len(data)
    for i in range(L - window):
        # input, first 11 months
        x_sequence = data[i:i + window]
        # expect output, next 11 months (Eg. Use Jan to pred Tue)
        y_sequence = data[i + 1: i + window + 1]
        # total is 12 months, use generator to get multiple groups
        yield x_sequence, y_sequence


def prepare_datasets(path):
    data = read_csv(path, parse_dates=[0], index_col=0)
    # data.index = data.index.to_period('M')
    data.Robberies = data.Robberies.astype(np.float32)
    # Reserve last 24 months as test set:
    train_data = data.Robberies[:-24].to_numpy().reshape(-1, 1)
    valid_data = data.Robberies[-24:].to_numpy().reshape(-1, 1)
    # scale data
    t_scaler = MinMaxScaler(feature_range=(-1, 1))
    v_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = t_scaler.fit_transform(train_data)
    valid_data = v_scaler.fit_transform(valid_data)

    # convert training data to tensor
    train_data = torch.tensor(train_data, device=torch.device('cuda'), dtype=torch.float32)
    valid_data = torch.tensor(valid_data, device=torch.device('cuda'), dtype=torch.float32)

    # Create validation set:
    # prepare for inputs:
    valid_x = valid_data[:-1]
    # prepare for expect outputs:
    valid_y = valid_data[1:]
    valid_data = (valid_x, valid_y)
    return train_data, valid_data, data, t_scaler, v_scaler


def train(model, epochs, train_set, valid_data=None, lr=0.001, print_every=100):
    # mean squared error (squared L2 norm)
    criterion = nn.MSELoss()
    # optimizer is Adam
    # optimizer parameters are weights and biases
    opt = optim.Adam(model.parameters(), lr=lr)
    # record for loss
    train_loss = []
    valid_loss = []

    for e in range(epochs):
        # No need for hidden state for the first time
        hs = None
        # total loss
        t_loss = 0
        for x, y in get_batches(train_set, 12):
            # x is input, y is expected output
            # do not accumulate grad between different batches
            opt.zero_grad()
            # Create batch_size dimension, group 11 months to one
            x = x.unsqueeze(0)
            # input to model and get output
            out, hs = model(x, hs)
            # strip out h.data and list to tuple
            hs = tuple([h.data for h in hs])
            # calculate loss
            loss = criterion(out, y)
            # back propagation, calculate grads
            loss.backward()
            # update weights according to grads
            opt.step()
            # get the float type of loss
            t_loss += loss.item()

        if valid_data is not None:
            # close dropout layers, batchNorm layers for eval
            model.eval()
            # the same with training part
            val_x, val_y = valid_data
            val_x = val_x.unsqueeze(0)
            # no need for hidden states output
            preds, _ = model(val_x, hs)
            v_loss = criterion(preds, val_y)
            valid_loss.append(v_loss.item())
            # open closed layers for continue learning
            model.train()

        train_loss.append(np.mean(t_loss))

        if e % print_every == 0:
            print(f'Epoch {e}:\nTraining Loss: {train_loss[-1]}')
            if valid_data is not None:
                print(f'Validation Loss: {valid_loss[-1]}')

    plt.figure(figsize=[8., 6.])
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.show()


def visualize(model, train_set, valid_set, origin, t_scaler, v_scaler):
    # get predictions on training set
    hs = None

    train_preds, hs = model(train_set.unsqueeze(0), hs)
    train_preds = train_preds.cpu()
    train_preds = t_scaler.inverse_transform(train_preds.detach())

    # Get predictions on validation data
    valid_x, valid_y = valid_set
    valid_preds, hs = model(valid_x.unsqueeze(0), hs)
    valid_preds = valid_preds.cpu()
    MinMaxScaler(feature_range=(-1, 1))
    valid_preds = v_scaler.inverse_transform(valid_preds.detach())
    train_time = origin.index[1:-23]
    valid_time = origin.index[-23:]
    plt.plot(train_time, train_preds.squeeze(), 'r--', label='Training Predictions', )
    plt.plot(valid_time, valid_preds.squeeze(), 'g--', label='Validation Predictions')
    plt.plot(origin.index, origin.Robberies.to_numpy(), label='Actual')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    input_size = 1
    hidden_size = 100
    num_layers = 1
    output_size = 1
    train_set, valid_set, origin, t, v = prepare_datasets('monthly-robberies.csv')
    # model = torch.load("a.pkl")
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.cuda(0)
    train(model, 1000, train_set, lr=0.0005, valid_data=valid_set)
    torch.save(model, "a.pkl")
