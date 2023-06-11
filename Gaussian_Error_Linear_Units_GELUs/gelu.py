import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from keras.datasets.mnist import load_data
sns.set_theme()

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 127.5 - 1
testX = np.float32(testX) / 127.5 - 1


def train(model, optimizer, loss_fct=torch.nn.NLLLoss(), nb_epochs=25):
    training_loss = []
    validation_loss = []
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

        # Validation
        model.train(False)
        log_prob = model(torch.from_numpy(testX).reshape(-1, 28 * 28).to(device))
        t_loss = loss_fct(log_prob, torch.from_numpy(testy).to(device))
        validation_loss.append(t_loss.item())
        model.train(True)

    return training_loss, validation_loss


class GELU(nn.Module):

    def __init__(self, alpha=1.):
        super(GELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / np.sqrt(2.)))


def _init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        module.weight.data = torch.nn.functional.normalize(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()


if __name__ == "__main__":
    dataset = DataLoader([[trainX[i], trainy[i]] for i in range(trainX.shape[0])], batch_size=64, shuffle=True)
    testing_dataset = DataLoader([[testX[i], testy[i]] for i in range(testX.shape[0])], batch_size=64, shuffle=True)

    device = 'cuda'
    model = torch.nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 10), nn.LogSoftmax(dim=-1)).to(device)
    model.apply(_init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_loss, validation_loss = train(model, optimizer, nb_epochs=150)
    plt.plot(training_loss, label='ReLU')
    plt.plot(validation_loss, label='ReLU', linestyle='--')

    model = torch.nn.Sequential(nn.Linear(28 * 28, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 128), GELU(), nn.Dropout(p=0.5),
                                nn.Linear(128, 10), nn.LogSoftmax(dim=-1)).to(device)
    model.apply(_init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    training_loss, validation_loss = train(model, optimizer, nb_epochs=150)
    plt.plot(training_loss, label='GELU')
    plt.plot(validation_loss, label='GELU', linestyle='--')

    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Cross Entropy Loss', fontsize=14)
    plt.savefig(f'Imgs/gelu.png')
