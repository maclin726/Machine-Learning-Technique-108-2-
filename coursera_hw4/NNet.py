import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, input_size, M, r):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, 1)
        nn.init.uniform_(self.fc1.weight, a=-r, b=r)
        nn.init.uniform_(self.fc2.weight, a=-r, b=r)

    def forward(self, x):
        output = self.fc1(x)
        output = torch.tanh(output)
        output = self.fc2(output)
        output = torch.tanh(output)
        return output.squeeze(1)

class p14Net(nn.Module):
    def __init__(self, input_size):
        super(p14Net, self).__init__()

        self.fc = nn.Sequential(
                    nn.Linear(input_size, 8),
                    nn.Tanh(),
                    nn.Linear(8, 3),
                    nn.Tanh(),
                    nn.Linear(3, 1),
                    nn.Tanh()
                 )
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)

        self.fc.apply(init_weights)

    def forward(self, x):
        output = self.fc(x)
        return output.squeeze(1)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = torch.FloatTensor(data[:, :-1])
        self.y = torch.FloatTensor(data[:, -1])
        self.dim = self.X.shape[1]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

with open('hw4_nnet_train.dat', 'r') as f:
    train_data = np.array([ list(map(float, line.strip().split(' '))) for line in f.readlines()])

with open('hw4_nnet_test.dat', 'r') as f:
    test_data = np.array([ list(map(float, line.strip().split(' '))) for line in f.readlines()])

def get_device(num='0'):
    ''' Get device (if GPU is available, use GPU) '''
    return ('cuda:' + num) if torch.cuda.is_available() else 'cpu'

device = get_device()

train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

num_steps = 50000
TIMES = 10

def train(model, lr):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    step = 0
    model.train()
    while step < num_steps:
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            predicts = model(inputs)
            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
        
        # if step % 10000 == 9999:
        #     print('loss:', running_loss)

    model.eval()
    correct = 0
    for x, y in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = torch.sign(model(x))
        correct += sum(y == pred.detach().cpu()).item()
    
    print(f'error: {1-correct/len(test_dataset):.3f}')
    return 1 - correct / len(test_dataset)


def p14():
    error = []
    for times in range(TIMES):
        model = p14Net(input_size=train_dataset.dim).to(device)
        error.append(train(model, lr=0.01))
    print(sum(error) / TIMES)

def p11():
    M = [1, 6, 11, 16, 21]
    M_error = []

    for m in M:
        print(f'M = {m}')
        error = []
        for times in range(TIMES):
            model = Net(input_size=train_dataset.dim, M=m, r=0.1).to(device)
            error.append(train(model, lr=0.1))
        M_error.append(sum(error) / TIMES)
    
    print(M_error)
    # [0.2584000000000001, "0.03640000000000003", 0.03640000000000003, 0.037600000000000036, 0.03640000000000003]

def p12():
    R = [0, 0.001, 0.1, 10, 1000]
    R_error = []

    for r in R:
        print(f'R = {r}')
        error = []
        for times in range(TIMES):
            model = Net(input_size=train_dataset.dim, M=3, r=r).to(device)
            error.append(train(model, lr=0.1))
        R_error.append(sum(error) / TIMES)
    
    print(R_error)
    # [0.03600000000000003, 0.036800000000000034, 0.03600000000000003, 0.3172, 0.48600000000000004]


def p13():
    LR = [0.001, 0.01, 0.1, 1, 10] 
    LR_error = []

    for lr in LR:
        print(f'LR = {lr}')
        error = []
        for times in range(TIMES):
            model = Net(input_size=train_dataset.dim, M=3, r=0.1).to(device)
            error.append(train(model, lr=lr))
        LR_error.append(sum(error) / TIMES)
    
    print(LR_error)
    # [0.18119999999999997, "0.03600000000000003", 0.03640000000000003, 0.03920000000000003, 0.48360000000000003]

p11()
p12()
p13()
p14()