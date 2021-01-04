import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import itertools


class Net(nn.Module):
    """Neural network that has autoencoder structure."""
    def __init__(self) -> None:
        """Initialize fields."""
        super(Net, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 4, 3)
        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)
        self.fcc1 = nn.Linear(1300, 64)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 8, 3)
        self.t_conv2 = nn.ConvTranspose2d(8, 1, 3)
        self.unpool1 = nn.MaxUnpool2d((2, 2))
        self.unpool2 = nn.MaxUnpool2d((2, 2))
        self.t_fcc1 = nn.Linear(64, 1300)

    def encode(self, x):
        """Encodes pictures."""
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
        return x, sizes, indices

    def decode(self, x, sizes, indices):
        """Decodes encoded pictures."""
        x = self.t_fcc1(x)
        x = x.reshape(sizes[2])
        x = self.unpool2(x, indices[1], output_size=sizes[1])
        x = F.relu(self.t_conv1(x))
        x = self.unpool1(x, indices[0], output_size=sizes[0])
        x = torch.sigmoid(self.t_conv2(x))
        return x

    def forward(self, x):
        """Sequentially encodes and decodes pictures."""
        x, sizes, indices = self.encode(x)
        x = self.decode(x, sizes, indices)
        return x


class Autoencoder:
    """Encodes and decodes play pictures."""
    def __init__(self, device="cuda") -> None:
        """Sets random seeds and fields."""
        self.set_seeds()
        self.device = device

    def set_seeds(self) -> None:
        """Sets various seeds for determinism when training autoencoder."""
        torch.manual_seed(0)
        np.random.seed(0)
        # torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)

    def load_data(self, team: str, batch: int, shuff: bool) -> tuple:
        """Creates a DataLoader object for the autoencoder."""
        transform_sequence = [transforms.Grayscale(num_output_channels=1),
                              transforms.ToTensor()]
        grey_transform = transforms.Compose(transform_sequence)
        data_folder = datasets.ImageFolder('../data/play_pics/' + team,
                                           transform=grey_transform)
        data_loader = torch.utils.data.DataLoader(data_folder,
                                                  batch_size=batch,
                                                  shuffle=shuff)
        return data_folder, data_loader

    def train(self, team: str, batch_size: int) -> None:
        """Trains the autoencoder and saves snapshots of NN occasionally."""
        _, data_loader = self.load_data(team, batch_size, True)
        net = Net().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        for epoch in range(10):
            running_loss = []
            for batch_idx, (data, _) in enumerate(data_loader):
                target_new_data = data.detach().to(self.device)
                optimizer.zero_grad()
                outputs = net(target_new_data)
                loss = criterion(outputs, target_new_data)
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
                if batch_idx % 1000 == 0:
                    print('Epoch: ', epoch,
                          ', Batch: ', batch_idx,
                          ', Loss: ', np.mean(running_loss))
            outpath = '../data/autoencoder/net_snapshots/' + \
                      team + '/epoch_' + str(epoch)
            torch.save(net.state_dict(), outpath)

    def show_pic(self, team: str, num_pics: int) -> None:
        """Shows a specified number of denoised pictures of a team."""
        _, data_loader = self.load_data(team, 1, True)
        model = Net()
        path = '../data/autoencoder/net_snapshots/' + \
               team + '/epoch_' + str(9)
        model.load_state_dict(torch.load(path))

        for data, _ in itertools.islice(data_loader, num_pics):
            plt.imshow(data.numpy()[0][0], cmap='gray')
            plt.show()
            output = model(torch.flatten(data, start_dim=1, end_dim=1))
            output.detach_()
            plt.imshow(output.reshape(108, 60).numpy(), cmap='gray')
            plt.show()

    def compress(self, team: str) -> None:
        """Compresses all images of a given team to numpy matrix."""
        data_folder, data_loader = self.load_data(team, 1, False)
        model = Net()
        path = '../data/autoencoder/net_snapshots/' + \
               team + '/epoch_' + str(9)
        model.load_state_dict(torch.load(path))
        compressions = []
        for idx, (data, _) in enumerate(data_loader):
            label = self.label_converter(data_folder.imgs[idx][0])
            output = model.encode(data)
            compression = output[0][0].detach().numpy()
            compressions.append(np.concatenate((label, compression)))
        outpath = '../data/autoencoder/compressions/' + team + '/compressions'
        np.save(outpath, compressions)

    @staticmethod
    def label_converter(label: str) -> list:
        """Gets game id and play id from label."""
        new_label = label[30:-4]
        game_id, play_id = new_label.split('-')
        return [int(game_id), int(play_id)]


if __name__ == '__main__':
    autoencoder = Autoencoder(device="cuda")
    autoencoder.train('def', 8)
    autoencoder.train('off', 8)
    autoencoder.compress('def')
    autoencoder.compress('off')
