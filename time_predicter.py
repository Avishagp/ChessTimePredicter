import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from data import data_parser
from dataset import ChessDataset
from tqdm import tqdm

NUMBER_OF_FEATURES = 5
NUMBER_OF_CLASSES = 16


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_size = NUMBER_OF_FEATURES
        self.fc0 = nn.Linear(NUMBER_OF_FEATURES, 10)
        self.fc1 = nn.Linear(10, NUMBER_OF_CLASSES)

    def forward(self, x):
        x = x.view(1, self.param_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)


def train(epoch, model, optimizer, train_loader):
    model.train()
    correct = 0
    for ep in range(0, epoch):
        correct = 0
        for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).cpu().sum()

        print('Epoch: {} \nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
              (ep, loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data.float())
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def predicton(test_x, model):
    f = open("model.txt", "w")
    for x in test_x:
        output = model(x)
        label = np.argmax(output.data)
        f.write(str(label.item()) + '\n')


def main():
    # Create dataset
    data_parser.parser(r"data/chess_data.txt")
    data = ChessDataset(data_parser.openings, data_parser.elo_dif, data_parser.higher_ranker, data_parser.utc_time,
                        data_parser.turns)

    # Split to training and testing
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    # Create train loader and test loader
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    # Model has as one row (looking at one turn) and 5 columns (as number of parameters)
    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Test and train
    train(10, model, optimizer, train_loader)
    test(model, test_loader)

    # Get test
    # loaded_test_x = np.loadtxt("test_x")
    # test_x = transform(loaded_test_x)[0].float()
    # predicton(test_x, model)


if __name__ == "__main__":
    main()
