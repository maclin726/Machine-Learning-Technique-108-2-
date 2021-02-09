import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy

# global variables
T = 5000
d_lst = [int(2**i) for i in range(1,8)]
ds = [i for i in range(1,8)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(in_features=kwargs["d0"], out_features=kwargs["d1"]).to(device)
        self.hidden = nn.Tanh().to(device)
        self.decoder = nn.Linear(in_features=kwargs["d1"], out_features=kwargs["d0"]).to(device)
        U = (6 / (1+kwargs['d0']+kwargs['d1'])) ** 0.5
        self.encoder.bias.data.uniform_(-U, U)
        self.encoder.weight.data.uniform_(-U, U)
        self.decoder.bias.data.uniform_(-U, U)
        if kwargs['tied'] == True:
            self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1)
        else:
            self.decoder.weight.data.uniform_(-U, U)

    def forward(self, x0):
        x1 = self.encoder(x0)
        x2 = self.hidden(x1)
        x3 = self.decoder(x2)
        return x3

with open('./zip.train', "r") as f:
    lines = f.readlines()
    X_train = np.array([ list(map(float, line.strip().split(' ')))[1:] for line in lines ])
    X_train_PCA = X_train.copy()
    X_train = torch.Tensor(X_train).to(device)

with open('./zip.test', "r") as f:
    lines = f.readlines()
    X_test = np.array([ list(map(float, line.strip().split(' ')))[1:] for line in lines ])
    X_test_PCA = X_test.copy()
    X_test = torch.Tensor(X_test).to(device)

def AE(tied):
    global d_lst, ds, T, device, X_train, X_test
    E_in, E_out = [], []
    for d in d_lst:
        model = AutoEncoder(d0 = 256, d1 = d, tied = tied)
        opt = optim.SGD(model.parameters(), lr=1e-1)
        criterion = nn.MSELoss(reduction='mean')
        for epoch in range(T):
            opt.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, X_train)
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, T, loss.item()))
        E_in.append(loss.item())
        outputs = model(X_test)
        loss = criterion(outputs, X_test)
        E_out.append(loss.item())
    return E_in, E_out

# Unconstained Autoencoder
E_in_11, E_out_12 = AE(False)

# Constrained Autoencoder
E_in_13, E_out_14 = AE(True)

def PCA(data, d):
    x_bar = np.mean(data, axis = 0)
    norm_data = data - x_bar
    matrix = norm_data.T @ norm_data
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    arg_sort = np.argsort(eigenvalues)[::-1]
    top_index = arg_sort[:d]
    top_vectors = eigenvectors[:,top_index]
    return top_vectors, x_bar

E_in_15, E_out_16 = [], []
for d in d_lst:
    W, x_bar = PCA(X_train_PCA, d)
    print(x_bar)
    X = ((W @ W.T @ (X_train_PCA - x_bar).T).T + x_bar)
    E_in_15.append(np.mean((X - X_train_PCA) ** 2))

    X = ((W @ W.T @ (X_test_PCA - x_bar).T).T + x_bar)
    E_out_16.append(np.mean((X - X_test_PCA) ** 2))

plt.figure(1)
plt.title(r'$E_{in}$ versus different $\tilde{d}$')
plt.plot(ds, E_in_11, label = r'$E_in$')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{in}$')
plt.savefig('p11.png', format='png')

plt.figure(2)
plt.title(r'$E_{out}$ versus different $\tilde{d}$')
plt.plot(ds, E_out_12, label = r'$E_out$')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{out}$')
plt.savefig('p12.png', format='png')

plt.figure(3)
plt.title(r'error versus different $\tilde{d}$')
plt.plot(ds, E_in_11, label = 'Unconstrained W')
plt.plot(ds, E_in_13, label = 'Constrained W')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{in}$')
plt.legend()
plt.savefig('p13.png', format='png')

plt.figure(4)
plt.title(r'error versus different $\tilde{d}$')
plt.plot(ds, E_out_12, label = 'Unconstrained W')
plt.plot(ds, E_out_14, label = 'Constrained W')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{out}$')
plt.legend()
plt.savefig('p14.png', format='png')

plt.figure(5)
plt.title(r'error versus different $\tilde{d}$')
plt.plot(ds, E_in_13, label = r'Constrained Autoencoder $E_{in}$')
plt.plot(ds, E_in_15, label = r'PCA $E_{in}$')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{in}$')
plt.legend()
plt.savefig('p15.png', format='png')

plt.figure(6)
plt.title(r'error versus different $\tilde{d}$')
plt.plot(ds, E_out_14, label = r'Unconstrained Autoencoder $E_{out}$')
plt.plot(ds, E_out_16, label = r'PCA $E_{out}$')
plt.xlabel(r'$\log_2 \tilde{d}$')
plt.ylabel(r'$E_{out}$')
plt.legend()
plt.savefig('p16.png', format='png')