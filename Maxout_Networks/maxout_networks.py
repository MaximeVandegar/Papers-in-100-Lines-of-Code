import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data
import seaborn as sns
sns.set_theme()

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255.
testX = np.float32(testX) / 255.


class Maxout(nn.Module):

    def __init__(self, din, dout, k):
        super(Maxout, self).__init__()

        self.net = nn.Linear(din, k * dout)
        self.k = k
        self.dout = dout

    def forward(self, x):
        return torch.max(self.net(x).reshape(-1, self.k * self.dout).reshape(-1, self.dout, self.k), dim=-1).values


def train(model, optimizer, loss_fct, nb_epochs, batch_size, trainX, trainy, testX, testy):

    training_loss = []
    testing_accuracy = []
    for epoch in tqdm(range(nb_epochs)):

        batch_training_loss = []

        # Sample batch
        idx = torch.randperm(trainX.shape[0])

        for indices in idx.chunk(int(np.ceil(trainX.shape[0] / batch_size))):
            x = trainX[indices].reshape(-1, 28 * 28)
            y = trainy[indices]

            log_prob = model(torch.from_numpy(x).to(device))
            loss = loss_fct(log_prob, torch.from_numpy(y).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_training_loss.append(loss.item())
        training_loss.append(np.mean(batch_training_loss))

        # Testing
        model.train(mode=False)
        log_prob = model(torch.from_numpy(testX.reshape(-1, 28 * 28)).to(device))
        testing_accuracy.append(
            (log_prob.argmax(-1) == torch.from_numpy(testy).to(device)).sum().item() / testy.shape[0])
        model.train(mode=True)

    return training_loss, testing_accuracy


if __name__ == "__main__":
    device = 'cuda'
    k = 4
    nb_epochs = 50
    batch_size = 128

    model = torch.nn.Sequential(nn.Dropout(p=0.4),  # See col. 2 p. 2
                                Maxout(28 * 28, 1200, k),
                                nn.Dropout(p=0.4),
                                Maxout(1200, 10, k),
                                nn.LogSoftmax(dim=-1)).to(device)
    loss_fct = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-6)

    training_loss, testing_accuracy = train(model, optimizer, loss_fct, nb_epochs, batch_size, trainX, trainy, testX,
                                            testy)

    ax = sns.lineplot(x=np.linspace(1, nb_epochs, nb_epochs), y=testing_accuracy)
    ax.set(xlabel='Epochs', ylabel='Testing accuracy')
    plt.savefig('Imgs/maxout_networks.png', bbox_inches='tight')
