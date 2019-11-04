import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class Ace_Dataset(Dataset):

    def __init__(self, ace_data, kp_data, x_frame, start, last):
        self.ace_data = ace_data
        self.kp_data = pd.read_csv(kp_data)
        self.start = start
        self.last = last
        self.x_frame = x_frame

        self.ace_data = self.ace_data.loc[(self.ace_data['year'] >= self.start) & (self.ace_data['year'] <= self.last)]
        self.ace_data = self.ace_data[['Vp','B_gsm_x','B_gsm_y','B_gsm_z','Bt']]

        self.kp_data['date'] = pd.to_datetime(self.kp_data['date'], format='%Y-%m-%d')
        self.kp_data['year'] = self.kp_data['date'].dt.year
        self.kp_data = self.kp_data.loc[(self.kp_data['year'] >= self.start) & (self.kp_data['year'] <= self.last)]
        self.kp_data = self.kp_data[['kp_0h', 'kp_3h', 'kp_6h', 'kp_9h', 'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h']]
        self.kp_data = np.array(self.kp_data.values).transpose(0,1)
        self.kp_data = self.kp_data.reshape(-1,1)



    def __len__(self):
        return len(self.kp_data)-1

    def __getitem__(self, idx):

        X = self.ace_data.iloc[self.x_frame * idx:(self.x_frame * idx) + self.x_frame].values
        y = self.kp_data[idx+1, 0]

        return X, y


# 모델 구성
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def make_regressor(self):
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1))
        return y_pred



def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    model.train()
    model.zero_grad()
    optimizer.zero_grad()


    train_loss = 0.0
    accuracy = 0.0
    for i, (X, y) in enumerate(trainloader):

        X = X.transpose(0, 1).float().to(args.device)
        y_true = y.float().to(args.device)


        model.zero_grad()
        optimizer.zero_grad()
        model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        y_pred = model(X)
        loss = loss_fn(y_pred.view(-1), y_true.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        accuracy += getAcc(y_pred, y_true)

    train_loss = train_loss / len(trainloader)
    accuracy = accuracy/len(trainloader)
    return model, train_loss,accuracy

def validate(model, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()


    val_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y.float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            loss = loss_fn(y_pred.view(-1), y_true.view(-1))


            val_loss += loss.item()
            accuracy += getAcc(y_pred, y_true)

    val_loss = val_loss / len(valloader)
    accuracy = accuracy/len(valloader)
    return val_loss, accuracy

def test(model, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()

    accuracy = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):

            X = X.transpose(0, 1).float().to(args.device)
            y_true = y.float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)

            accuracy += getAcc(y_pred, y_true)


    accuracy = accuracy/len(testloader)
    return accuracy

# 모델 평가(accuracy 방법)
def getAcc(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()
    correct = 0
    for i in range(len(perc_y_pred)):
        y__pred = perc_y_pred[i]
        y__true = perc_y_true[i]

        if  y__pred - int(y__pred) < 0.5:
            y__pred = int(y__pred)
        else:
            y__pred = int(y__pred) + 1

        if y__pred == y__true:
            correct += 1
    acc = correct/len(perc_y_pred)*100
    return acc


def experiment(partition, args):
    model = LSTM(args.input_dim, args.hid_dim, args.output_dim, args.n_layers, args.batch_size, args.dropout,
                 args.use_bn)
    model.to(args.device)
    print('cuda', torch.cuda.is_available())

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)


    for epoch in range(args.epoch):
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args)
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()


        print(
            'Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}, Acc(train/val) {:.2f}%/{:.2f}%. Took {:2.2f} sec'.format(epoch,
                                                                                                                train_loss,
                                                                                                                val_loss,
                                                                                                                train_acc,
                                                                                                                val_acc,
                                                                                                                te - ts))
    test_acc = test(model, partition, args)
    print('test_accuracy', test_acc)

# 모델 저장(h5로 저장)
    torch.save(model.state_dict(), args.model)

class getresult(Dataset):

    def __init__(self, data, re_y, x_frame):
        self.data = data
        self.re_y = re_y
        self.x_frame = x_frame

    def __len__(self):
        return 5840

    def __getitem__(self, idx):
        X = self.data.iloc[self.x_frame * idx:(self.x_frame * idx) + self.x_frame].values
        y = self.re_y

        return X, y