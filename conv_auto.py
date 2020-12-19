import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))

        return x


no_transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='data', train=True,
                                      download=False, transform=no_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)


the_net = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(the_net.parameters(), lr=0.001)

for epoch in range(50):
    running_loss = []
    for batch_idx, (data, _) in enumerate(trainloader):
        target_new_data = data.detach()
        optimizer.zero_grad()
        outputs = the_net(target_new_data)
        loss = criterion(outputs, target_new_data)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        if batch_idx % 1000 == 0:
            print('Epoch: ', epoch, ', Batch: ', batch_idx, ', Loss: ', np.mean(running_loss))
    torch.save(the_net.state_dict(), 'conv_autoencoder_' + str(epoch+1))


# model = Net()
# model.load_state_dict(torch.load('autoencoder_10'))
#
#
# for batch_idx, (data, lab) in enumerate(trainloader):
#     print(lab)
#     plt.imshow(data.numpy()[0][0], cmap='gray')
#     plt.show()
#     output = model(torch.flatten(data, start_dim=1, end_dim=3))
#     output.detach_()
#     plt.imshow(output.reshape(28, 28).numpy(), cmap='gray')
#     plt.show()
#     break
