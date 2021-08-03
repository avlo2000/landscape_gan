import numpy as np
import torch
import utils
from torch import nn, optim
from tqdm import tqdm


def _get_output_shape(model, img_shape):
    zeros = torch.zeros(img_shape)
    zeros = zeros.unsqueeze(0)
    return model(zeros).data.shape[1:]


class AutoEncoder(nn.Module):
    def __init__(self, img_sizes=(512, 512), in_channels=3, latent_space_size: int = 50):
        super().__init__()

        kernels_start_power_of_2 = 4

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16, kernel_size=(3, 3)
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3)
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3)
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        img_shape = (in_channels, img_sizes[0], img_sizes[0])
        self.encoder_out_shape = _get_output_shape(self.encoder, img_shape)
        latent_space_in = np.prod(list(self.encoder_out_shape))
        latent_space_out = latent_space_in

        self.latent_space = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_space_in, latent_space_size),
            nn.Sigmoid(),
            nn.Linear(latent_space_size, latent_space_out),
            nn.Sigmoid(),
            nn.Unflatten(dim=1,
                         unflattened_size=self.encoder_out_shape)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(3, 3)
            ),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=(3, 3)
            ),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=3, kernel_size=(3, 3)
            ),
            nn.ReLU(True)
        )

    def forward(self, features):
        x = self.encoder(features)
        x = self.latent_space(x)
        x = self.decoder(x)
        return x


def train_model(train_x, test_x, train_y, test_y, model: nn.Module, epochs, batch_size=32, **kwargs):
    assert len(train_x) == len(train_y)
    train_size = len(train_x)
    test_size = len(test_x)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    running_loss = 0.0

    for epoch in range(epochs):
        permutation = torch.randperm(train_size)
        for i in tqdm(range(0, train_size, batch_size)):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x = torch.stack([torch.from_numpy(train_x[i]).float() for i in indices])
            batch_y = torch.stack([torch.from_numpy(train_y[i]).float() for i in indices])
            actual_y = model.forward(batch_x)
            loss = criterion(actual_y, batch_y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch " + str(epoch + 1) + " loss: " + str(running_loss))

        if 'vis_test' in kwargs and kwargs['vis_test']:
            tests = 5
            if 'vis_test_size' in kwargs:
                tests = kwargs['vis_test_size']
            test_inputs = torch.stack([torch.from_numpy(test_x[i]).float() for i in torch.randperm(test_size)[:tests]])
            test_outputs = model(test_inputs)
            for inp, out in zip(test_inputs, test_outputs):
                img_out = (np.transpose(out.detach().numpy()) * 255).astype('uint8')
                img_in = (np.transpose(inp.detach().numpy()) * 255).astype('uint8')
                utils.show_two_images(img_in, img_out)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
        }, 'model_backup.weights')

    torch.save(model.state_dict(), 'model.weights')