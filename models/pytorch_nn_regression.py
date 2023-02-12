import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np


class NN(nn.Module):
    def __init__(self, input_size, num_hidden) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 1)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def run_regression(train_x, train_target, test_x, test_target, learning_rate = 0.005, batch_size = 30, num_epochs = 50, loss_function = nn.MSELoss, num_hidden = 15, verbose=True):
    #Set device, run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    #Load Data
    train_x = torch.Tensor(train_x.reshape((train_x.shape[0], -1))).float()
    train_target = torch.tensor(train_target).float().squeeze()
    train_dataset = TensorDataset(train_x, train_target)

    test_x = torch.Tensor(test_x.reshape((test_x.shape[0], -1))).float()
    test_target = torch.tensor(test_target).float().squeeze()
    test_dataset = TensorDataset(test_x, test_target)

    #Hyperparameters
    input_size = train_x.size()[1]

    #Load dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the network
    model = NN(input_size=input_size, num_hidden=num_hidden).to(device=device)


    #Loss function and optimizer
    criterion = loss_function()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_accuracy = []
    test_accuracy = []
    #Train network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device).squeeze()

            data = data.reshape(data.shape[0], -1)

            #forward
            scores = model(data).float().squeeze()
            loss = criterion(scores, targets).float()

            # print(loss)

            #backward
            optimizer.zero_grad()
            loss.backward()


            #gradient descent
            optimizer.step()
        if verbose:
            print(f'Epoch {epoch}: accuracy on train data: {check_accuracy(train_loader, model, device):.2f}, accuracy on test data: {check_accuracy(test_loader, model, device):.2f}')

    
        train_accuracy.append(check_accuracy(train_loader, model, device))
        test_accuracy.append(check_accuracy(test_loader, model, device))
    
    scores = []
    mae = []
    with torch.no_grad():
        scores = model(test_x).squeeze().numpy()
        truth = test_target.squeeze().numpy()
        mae = np.abs(scores-truth)

        scores_train = model(train_x).squeeze().numpy()
        truth_train = train_target.squeeze().numpy()
        mae_train = np.abs(scores_train-truth_train)

    return (train_accuracy, test_accuracy, scores, mae_train, mae, scores_train-truth_train, scores-truth)



def check_accuracy(loader, model, device):
    mae = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for (x,y) in loader:
            x = x.to(device=device)
            y= y.to(device=device)

            x = x.reshape(x.shape[0], -1)
            scores = model(x).squeeze()
            # print(scores-y)

            # _, predictions = scores.max(1)
            # print(scores.size())

            diff = nn.L1Loss()(scores, y)
            mae += diff
            # print(diff)
            num_samples += 1#predictions.size(0)

        return float(mae)/float(num_samples)