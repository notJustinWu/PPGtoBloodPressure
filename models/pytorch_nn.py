import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class NN(nn.Module):
    def __init__(self, input_size, num_classes, num_hidden) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, num_classes)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def run_classification(train_x, train_target, test_x, test_target, num_classes, learning_rate = 0.005, batch_size = 30, num_epochs = 20, loss_function = nn.CrossEntropyLoss, num_hidden = 20, verbose=True):
    #Set device, run on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    #Load Data
    train_x = torch.Tensor(train_x.reshape((train_x.shape[0], -1)))
    train_target = torch.tensor(train_target)
    train_dataset = TensorDataset(train_x, train_target)

    test_x = torch.Tensor(test_x.reshape((test_x.shape[0], -1)))
    test_target = torch.tensor(test_target)
    test_dataset = TensorDataset(test_x, test_target)

    #Hyperparameters
    input_size = train_x.size()[1]

    #Load dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the network
    model = NN(input_size=input_size, num_classes=num_classes, num_hidden=num_hidden).to(device=device)


    #Loss function and optimizer
    criterion = loss_function()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    #Train network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            data = data.reshape(data.shape[0], -1)

            #forward
            scores = model(data)
            loss = criterion(scores, targets)

            #backward
            optimizer.zero_grad()
            loss.backward()


            #gradient descent
            optimizer.step()
        if verbose:
            print(f'Epoch {epoch}: accuracy on train data: {check_accuracy(train_loader, model, device):.2f}, accuracy on train data: {check_accuracy(test_loader, model, device):.2f}')

    
    train_accuaracy = check_accuracy(train_loader, model, device)
    test_accuracy = check_accuracy(test_loader, model, device)
    print("############")
    print("############")
    print(f'Training complete: final accuracy on train data {train_accuaracy:.2f}, final accuracy of test data {test_accuracy:.2f}')

    with torch.no_grad():
        scores = model(test_x).max(1)[1].squeeze().numpy()

        scores_train = model(train_x).max(1)[1].squeeze().numpy()

        print(scores)

        return scores_train, scores



def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for (x,y) in loader:
            x = x.to(device=device)
            y= y.to(device=device)

            x = x.reshape(x.shape[0], -1)
            scores = model(x)

            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        return float(num_correct)/float(num_samples)*100