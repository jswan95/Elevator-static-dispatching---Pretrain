import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
traffic_pattern = 'DownPeak'
num_epochs = 200
learning_rate = 1e-4
batch_size = 128
criterion = nn.CrossEntropyLoss()


class ActorNetwork(nn.Module):
    def __init__(self, action_size):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(26, 64, kernel_size=5, stride=1, padding=2),
                                   nn.Tanh())  # Output shape: (batch_size, 64, 20)
        self.pooling1 = nn.MaxPool1d(kernel_size=2)  # Output shape: (batch_size, 64, 10)
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
                                   nn.Tanh())  # Output shape: (batch_size, 128, 10)
        self.pooling2 = nn.MaxPool1d(kernel_size=2)  # Output shape: (batch_size, 128, 5)
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=5, stride=1), nn.Tanh())  # Output shape: (batch_size, 256, 1)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, action_size))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.pooling2(out)
        out = self.conv3(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()  # sample an action


class PretrainData(Dataset):
    def __init__(self, x_path, y_path):
        # Data loading
        self.state = torch.from_numpy(np.loadtxt(x_path).reshape(-1, 26, 20)).float().to(device)
        self.y_labels = torch.from_numpy(np.loadtxt(y_path)).long().to(device)
        self.num_samples = len(self.y_labels)

    def __getitem__(self, index):
        return self.state[index], self.y_labels[index]

    def __len__(self):
        return self.num_samples


# Initialize wandb logger
wandb.init(project=f'Static dispatching-policy net pretraining {traffic_pattern}', name='pretrain_policy', mode='disabled')

# Load train and test datasets
train_dataset = PretrainData(x_path='train_data/state.txt', y_path='train_data/car_label.txt')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = PretrainData(x_path='test_data/state.txt', y_path='test_data/car_label.txt')
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize actor network and optimizer
policy_net = ActorNetwork(action_size=4).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, cooldown=5, min_lr=1e-6)


# Train and validate functions
def train(model, criterion, optimizer, data_loader):
    model.train()
    epoch_loss = []
    for i, (states, y_labels) in enumerate(data_loader):
        outputs = model(states)
        loss = criterion(outputs, y_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if (i + 1) % 50 == 0:
            wandb.log({'loss': loss.item()})
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss


def validate(model, scheduler, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for states, y_labels in data_loader:
            outputs = model(states)
            _, predicted = torch.max(outputs.data, 1)
            total += y_labels.size(0)
            correct += (predicted == y_labels).sum().item()
    accuracy = 100 * correct / total
    scheduler.step(accuracy)
    return accuracy


# Training loop
accuracy = 0
os.makedirs('result_folder', exist_ok=True)

for epoch in range(1, num_epochs + 1):
    train_epoch_loss = train(model=policy_net, criterion=criterion, optimizer=optimizer, data_loader=train_loader)
    temp_accuracy = validate(model=policy_net, scheduler=scheduler, data_loader=test_loader)
    print(f'Epoch {epoch}/{num_epochs},Train epoch loss {train_epoch_loss}, Test accuracy {temp_accuracy}')
    wandb.log({'Train epoch loss': train_epoch_loss, 'Test epoch accuracy': temp_accuracy, 'Epoch': epoch})
    if accuracy < temp_accuracy:
        torch.save(policy_net.state_dict(), f'result_folder/saved_agent_{epoch}.pt')
        accuracy = temp_accuracy

# Finish logging with wandb
wandb.finish()
