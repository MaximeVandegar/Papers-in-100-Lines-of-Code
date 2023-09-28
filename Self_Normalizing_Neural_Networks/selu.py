import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from keras.datasets.mnist import load_data
from sklearn.preprocessing import StandardScaler

sns.set_theme()

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
s = StandardScaler()
s.fit(trainX.reshape(-1, 28 * 28))
trainX = s.transform(trainX.reshape(-1, 28 * 28)).astype(np.float32)
testX = s.transform(testX.reshape(-1, 28 * 28)).astype(np.float32)


class SELU(nn.Module):

    def __init__(self):
        super(SELU, self).__init__()

        self.alpha = 1.6732632423543772848170429916717
        self.lambda_ = 1.0507009873554804934193349852946

    def forward(self, x):
        return self.lambda_ * (torch.maximum(torch.tensor([0], device=x.device), x
                                             ) + torch.minimum(torch.tensor([0], device=x.device),
                                                               self.alpha * (torch.exp(x) - 1)))


class SNN(nn.Module):

    def __init__(self, input_dim=28 * 28, hidden_dim=28 * 28, output_dim=10, depth=8):
        super(SNN, self).__init__()

        model = []
        for _ in range(depth):
            model += [nn.Linear(input_dim, hidden_dim), SELU()]
        model += [nn.Linear(hidden_dim, output_dim), nn.LogSoftmax(dim=-1)]
        self.network = nn.Sequential(*model)

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):

    def __init__(self, input_dim=28 * 28, hidden_dim=28 * 28, output_dim=10, depth=8):
        super(MLP, self).__init__()

        model = []
        for _ in range(depth):
            model += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), torch.nn.BatchNorm1d(hidden_dim)]
        model += [nn.Linear(hidden_dim, output_dim), nn.LogSoftmax(dim=-1)]
        self.network = nn.Sequential(*model)

    def forward(self, x):
        return self.network(x)


def train(model, optimizer, dataset, loss_fct=torch.nn.NLLLoss(), device='cpu', nb_epochs=25):
    training_loss = []
    for _ in tqdm(range(nb_epochs)):
        batch_loss = []
        for batch in dataset:
            x, y = batch
            log_prob = model(x.to(device).reshape(-1, 28 * 28))
            loss = loss_fct(log_prob, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        training_loss.append(np.mean(batch_loss))
    return training_loss


def _init_weights(module, init_gain=1.):
    if isinstance(module, nn.Linear):

        module.weight.data.normal_(mean=0.0, std=np.sqrt(init_gain / module.weight.data.shape[0]))
        if module.bias is not None:
            module.bias.data.zero_()


if __name__ == "__main__":
    device = 'cuda'
    nb_epochs = 2000
    dataset = DataLoader([[trainX[i], trainy[i]] for i in range(trainX.shape[0])], batch_size=128, shuffle=True)

    training_loss_snn = {}
    for depth in [8, 16, 32]:
        model = SNN(depth=depth).to(device)
        model.apply(_init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        training_loss_snn[depth] = train(model, optimizer, dataset, device=device, nb_epochs=nb_epochs)

    training_loss_mlp = {}
    for depth in [8, 16, 32]:
        model = MLP(depth=depth).to(device)
        model.apply(lambda x: _init_weights(x, init_gain=2.))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        training_loss_mlp[depth] = train(model, optimizer, dataset, device=device, nb_epochs=nb_epochs)

    fontsize = 14
    plt.plot(training_loss_mlp[8], color=plt.cm.bwr([0.05]), label='BatchNorm Depth 8')
    plt.plot(training_loss_mlp[16], color=plt.cm.bwr([0.15]), label='BatchNorm Depth 16')
    plt.plot(training_loss_mlp[32], color=plt.cm.bwr([0.25]), label='BatchNorm Depth 32')
    plt.plot(training_loss_snn[8], color=plt.cm.bwr([0.75]), label='SNN Depth 8')
    plt.plot(training_loss_snn[16], color=plt.cm.bwr([0.85]), label='SNN Depth 16')
    plt.plot(training_loss_snn[32], color=plt.cm.bwr([0.95]), label='SNN Depth 32')
    plt.yscale("log")
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Training Error", fontsize=fontsize)
    plt.legend()
    plt.savefig('Imgs/snn.png', bbox_inches='tight')
