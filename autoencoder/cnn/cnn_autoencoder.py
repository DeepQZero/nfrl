import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import PIL.Image as Image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 4, 3)
        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)
        self.fcc1 = nn.Linear(1300, 512)
        self.fcc2 = nn.Linear(512, 128)
        self.fcc3 = nn.Linear(128, 64)
        self.fcc4 = nn.Linear(64, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 3)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 3)
        self.unpool1 = nn.MaxUnpool2d((2, 2))
        self.unpool2 = nn.MaxUnpool2d((2, 2))
        self.t_fcc4 = nn.Linear(2, 64)
        self.t_fcc3 = nn.Linear(64, 128)
        self.t_fcc2 = nn.Linear(128, 512)
        self.t_fcc1 = nn.Linear(512, 1300)

    def encode(self, x):
        sizes = []
        indices = []
        x = F.relu(self.conv1(x))
        sizes.append(x.size())
        x, indices1 = self.pool1(x)
        indices.append(indices1)
        x = F.relu(self.conv2(x))
        sizes.append(x.size())
        x, indices2 = self.pool2(x)
        indices.append(indices2)
        sizes.append(x.size())
        x = x.reshape(x.size()[0], -1)
        x = self.fcc1(x)
        x = self.fcc2(x)
        x = self.fcc3(x)
        x = self.fcc4(x)
        return x, sizes, indices

    def decode(self, x, sizes, indices):
        x = self.t_fcc4(x)
        x = self.t_fcc3(x)
        x = self.t_fcc2(x)
        x = self.t_fcc1(x)
        x = x.reshape(sizes[2])
        x = self.unpool2(x, indices[1], output_size=sizes[1])
        x = F.relu(self.t_conv1(x))
        x = self.unpool1(x, indices[0], output_size=sizes[0])
        x = torch.sigmoid(self.t_conv2(x))
        return x

    def forward(self, x):
        x, sizes, indices = self.encode(x)
        x = self.decode(x, sizes, indices)
        return x


no_transform = transforms.Compose(
    [transforms.ToTensor()])

grey_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])

try_set = datasets.ImageFolder('../cnn/data/defense', transform=grey_transform)
print(try_set.imgs[1])

data_loader = torch.utils.data.DataLoader(try_set, batch_size=1, shuffle=False)

# device = "cuda"
# the_net = ConvAutoencoder().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(the_net.parameters(), lr=1e-3)
#
# for epoch in range(50):
#     running_loss = []
#     for batch_idx, (data, _) in enumerate(data_loader):
#         target_new_data = data.detach().to(device)
#         optimizer.zero_grad()
#         outputs = the_net(target_new_data)
#         loss = criterion(outputs, target_new_data)
#         loss.backward()
#         optimizer.step()
#
#         running_loss.append(loss.item())
#         if batch_idx % 1000 == 0:
#             print('Epoch: ', epoch, ', Batch: ', batch_idx, ', Loss: ', np.mean(running_loss))
#     torch.save(the_net.state_dict(), 'snapshots/defense/conv_autoencoder_' + str(epoch+1))


# model = ConvAutoencoder()
# model.load_state_dict(torch.load('snapshots/defense/conv_autoencoder_30'))
#
# for batch_idx, (data, lab) in enumerate(data_loader):
#     print(lab)
#     plt.imshow(data.numpy()[0][0], cmap='gray')
#     plt.show()
#     output = model(torch.flatten(data, start_dim=1, end_dim=1))
#     output.detach_()
#     plt.imshow(output.reshape(108, 60).numpy(), cmap='gray')
#     plt.show()
#     break


model = ConvAutoencoder()
model.load_state_dict(torch.load('snapshots/defense/conv_autoencoder_30'))

xs, ys = [], []



# for batch_idx, (data, lab) in enumerate(data_loader, 0):
#     plt.imshow(data.numpy()[0][0], cmap='gray')
#     image = torch.flatten(data, start_dim=1, end_dim=1)
#     output = model.encode(image)
#     xs.append(output[0][0][0].item())
#     ys.append(output[0][0][1].item())
#
# print(min(xs), max(xs))
# print(min(ys), max(ys))

# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

# datasets.
# image = Image.open('data/defense/def/2018090600-75.png')
# # image.show()
# image = grey_transform(image)



# imsize = 256
# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
#
# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     image = loader(image).float()
#     image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU
#
# image = image_loader(PATH TO IMAGE)
