import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6480, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc8 = nn.Linear(2048, 6480)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def decode(self, x):
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# no_transform = transforms.Compose([transforms.ToTensor()])

grey_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])

try_set = datasets.ImageFolder('../cnn/data/defense', transform=grey_transform)
data_loader = torch.utils.data.DataLoader(try_set, batch_size=1, shuffle=True)

device = "cuda"
the_net = Net()
the_net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(the_net.parameters(), lr=1e-4)

# for epoch in range(50):
#     running_loss = []
#     for batch_idx, (data, _) in enumerate(data_loader):
#         new_data = torch.flatten(data, start_dim=1, end_dim=3)
#         target_new_data = new_data.detach().to(device)
#         optimizer.zero_grad()
#         outputs = the_net(target_new_data)
#         loss = criterion(outputs, target_new_data)
#         loss.backward()
#         optimizer.step()
#
#         running_loss.append(loss.item())
#         if batch_idx % 1000 == 0:
#             print('Epoch: ', epoch, ', Batch: ', batch_idx, ', Loss: ', np.mean(running_loss))
#     torch.save(the_net.state_dict(), 'snapshots/autoencoder_' + str(epoch+1))


model = Net()
model.load_state_dict(torch.load('snapshots/autoencoder_49'))


for batch_idx, (data, lab) in enumerate(data_loader):
    print(lab)
    print(data.shape)
    print(torch.min(data), torch.max(data))
    plt.imshow(data.numpy()[0][0], cmap='gray')
    plt.show()
    output = model(torch.flatten(data, start_dim=1, end_dim=3))
    output.detach_()
    print(output.shape)
    plt.imshow(output.reshape(108, 60).numpy(), cmap='gray')
    plt.show()
    break
