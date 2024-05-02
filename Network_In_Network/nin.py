import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data
import seaborn as sns
sns.set_theme()

# load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 127.5 - 1.
testX = np.float32(testX) / 127.5 - 1.


class NiN(nn.Module):

    def __init__(self):
        super(NiN, self).__init__()

        conv1 = nn.Conv2d(1, 96, 5, padding=2)
        nn.init.normal_(conv1.weight, mean=0.0, std=0.05)
        cccp1 = nn.Conv2d(96, 64, 1)
        nn.init.normal_(cccp1.weight, mean=0.0, std=0.05)
        cccp2 = nn.Conv2d(64, 48, 1)
        nn.init.normal_(cccp2.weight, mean=0.0, std=0.05)
        conv2 = nn.Conv2d(48, 128, 5, padding=2)
        nn.init.normal_(conv2.weight, mean=0.0, std=0.05)
        cccp3 = nn.Conv2d(128, 96, 1)
        nn.init.normal_(cccp3.weight, mean=0.0, std=0.05)
        cccp4 = nn.Conv2d(96, 48, 1)
        nn.init.normal_(cccp4.weight, mean=0.0, std=0.05)
        conv3 = nn.Conv2d(48, 128, 5, padding=2)
        nn.init.normal_(conv3.weight, mean=0.0, std=0.05)
        cccp5 = nn.Conv2d(128, 96, 1)
        nn.init.normal_(cccp5.weight, mean=0.0, std=0.05)
        cccp6 = nn.Conv2d(96, 10, 1)
        nn.init.normal_(cccp6.weight, mean=0.0, std=0.05)

        self.model = nn.Sequential(conv1, nn.ReLU(),
                                   cccp1, nn.ReLU(),
                                   cccp2, nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2, padding=1),
                                   nn.Dropout(p=0.5),
                                   conv2, nn.ReLU(),
                                   cccp3, nn.ReLU(),
                                   cccp4, nn.ReLU(),
                                   nn.MaxPool2d(3, stride=2, padding=1),
                                   nn.Dropout(p=0.5),
                                   conv3, nn.ReLU(),
                                   cccp5, nn.ReLU(),
                                   cccp6,
                                   torch.nn.AvgPool2d(7, stride=1, padding=0))
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        log_prob = self.logsoftmax(self.model(x).squeeze(-1).squeeze(-1))
        return log_prob


def train(model, optimizer, scheduler, loss_fct, batch_size, trainX, trainy, 
          testX, testy, device):

    testing_accuracy = []
    initial_lr = optimizer.param_groups[0]['lr']
    while (optimizer.param_groups[0]['lr'] > (0.01 * initial_lr + 1e-15)):

        # Sample batch
        idx = torch.randperm(trainX.shape[0])

        train_batch_accuracy = 0.
        for indices in idx.chunk(int(np.ceil(trainX.shape[0] / batch_size))):
            x = trainX[indices]
            y = trainy[indices]

            log_prob = model(torch.from_numpy(x).unsqueeze(1).to(device))
            loss = loss_fct(log_prob, torch.from_numpy(y).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_accuracy += ((log_prob.argmax(-1) == torch.from_numpy(
                y).to(device)).sum().item() / y.shape[0])
        scheduler.step(train_batch_accuracy)

        # Testing
        model.eval()
        log_prob = model(torch.from_numpy(testX).unsqueeze(1).to(device))
        testing_accuracy.append((log_prob.argmax(-1) == torch.from_numpy(
            testy).to(device)).sum().item() / testy.shape[0])
        model.train()

    return testing_accuracy


if __name__ == "__main__":
    device = 'cuda'
    model = NiN().to(device)
    loss_fct = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5,
                          momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    testing_accuracy = train(model, optimizer, scheduler, loss_fct, 128,
                             trainX, trainy, testX, testy, device)

    plt.plot((1 - np.array(testing_accuracy)) * 100)
    plt.gca().set_ylim([0.47 - 0.15, 0.47 + 0.45])
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Test Error (%)', fontsize=13)
    plt.savefig('Imgs/nin.png', bbox_inches='tight')
    plt.close()
