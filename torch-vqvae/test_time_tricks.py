import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from modules import VectorQuantizedVAE, to_scalar, TestTimeVQVAE
from datasets import MiniImagenet
import matplotlib.pyplot as plt
import argparse
import os
import multiprocessing as mp
from tensorboardX import SummaryWriter

def check_membership(z_set, trial):

    for each in z_set:

        if torch.all(trial == each):
            return False


    return True

def viz_img(tensor):


    plt.imshow(tensor.squeeze().permute(1, 2, 0))
    plt.show()


def train_test_time_max(label, model, optimizer, loss):

    z_set = list()
    model.train()
    indices = model.initial_indices.flatten()
    step = 0
    while check_membership(z_set, indices):

        z_set.append(indices)
        optimizer.zero_grad()
        x_tilde, _, _, indices = model()

        calc_loss = loss(x_tilde, label)
        calc_loss.backward()

        current_loss = calc_loss.item()
        same_update = 0
        while calc_loss == current_loss:

            if same_update > 1000:
                return -1

            current_grad = torch.clone(model.optimize_latents.grad)
            normed_values = current_grad.norm(dim=1).squeeze()
            max_x, max_y = torch.where(normed_values == torch.max(normed_values))
            mask = torch.zeros_like(current_grad)
            mask[:, :, max_x, max_y] = 1.0
            model.optimize_latents.grad = mask * current_grad

            optimizer.step()
            x_tilde, _, _, indices = model()
            calc_loss = loss(x_tilde, label)

            same_update += 1

        viz_img(x_tilde.detach())
        step += 1


def train_test_time(label, model, optimizer, loss):

    z_set = list()
    model.train()
    indices = model.initial_indices.flatten()
    step = 0
    while check_membership(z_set, indices):

        z_set.append(indices)
        optimizer.zero_grad()
        x_tilde, _, _, indices = model()
        viz_img(x_tilde.detach())
        calc_loss = loss(x_tilde, label)
        calc_loss.backward()

        current_loss = calc_loss.item()
        same_update = 0
        while calc_loss == current_loss:

            if same_update > 1000:
                return -1

            optimizer.step()
            x_tilde, _, _, indices = model()
            calc_loss = loss(x_tilde, label)

            same_update += 1

        viz_img(x_tilde.detach())
        step += 1

dataset = 'miniimagenet'
data_folder = '/proj/vondrick3/ishaan/old/miniimagenet'

torch.cuda.manual_seed(0)
np.random.seed(0)


# Define the train & test datasets
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Define the train, valid & test datasets
train_dataset = MiniImagenet(data_folder, train=True,
                             download=True, transform=transform)
valid_dataset = MiniImagenet(data_folder, valid=True,
                             download=True, transform=transform)
test_dataset = MiniImagenet(data_folder, test=True,
                            download=True, transform=transform)
num_channels = 3
num_workers = mp.cpu_count() - 1
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=128, shuffle=False,
                                           num_workers=num_workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=128, shuffle=False, drop_last=True,
                                           num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=16, shuffle=False)


fixed_images, _ = next(iter(test_loader))

source = fixed_images[0][None, :]
target = fixed_images[12][None, :]

viz_img(source)
viz_img(target)

model_path = '/proj/vondrick3/ishaan/pytorch-vqvae/models/models/vqvae_imagenet/best.pt'
model = TestTimeVQVAE(model_path, source, 3, 256, 512)

#model_other = VectorQuantizedVAE(1,256,512)


#model_other(fixed_images)


loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


train_test_time(target, model, optimizer, loss)

#train_test_time_max(target, model, optimizer, loss)
