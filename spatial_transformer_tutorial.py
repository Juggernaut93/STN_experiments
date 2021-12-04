# -*- coding: utf-8 -*-
"""
Based on the "Spatial Transformer Networks Tutorial"
by Ghassen HAMROUNI <https://github.com/GHamrouni>
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
from CoordConv import CoordConv
import random
import argparse

# Reproducibility, see https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
# We can't enable this option because the grid_sampler_2d_backward_cuda
# function, used to propagate the gradients for the grid_sample
# operation, does not have a deterministic implementation in pyTorch
#torch.use_deterministic_algorithms(True)

# Device selection, prefer GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, use_coord=None, use_radius=False):
        """
        Parameters
        ----------
        use_coord : str, optional
            Whether CoordConv should replace Conv (default is None).
            'stn' indicates that only the STN module should use CoordConv.
            'all' indicates that all Conv modules should be replaced.
        use_radius: bool
            Whether CoordConv should add a third channel with radial
            information (default is False).
            Ignored if use_coord is None.
        
        Raises
        ------
        ValueError
            If use_coord argument value is invalid.
        """
        super(Net, self).__init__()
        
        if use_coord not in (None, 'stn', 'all'):
            raise ValueError("use_coord must be either None, 'stn' or 'all'")
        
        # CoordConv initializer proxy function, so we can easily select the correct module
        # and use it throughout the network without bothering
        cconv = lambda *vargs, **kwargs: CoordConv(*vargs, **kwargs, with_r=use_radius)
        # selecting the correct layer based on use_coord argument
        main_conv = cconv if use_coord == 'all' else nn.Conv2d # use CoordConv outside of STN only if use_coord is 'all'
        stn_conv = cconv if use_coord is not None else nn.Conv2d # use CoordConv in STN if use_coord is 'stn' or 'all'
        
        self.conv1 = main_conv(1, 10, kernel_size=5)
        self.conv2 = main_conv(10, 20, kernel_size=5)
        
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            stn_conv(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            stn_conv(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch, model, optimizer, train_loader):
    """
    Perform a training epoch on the provided model.
    
    Parameters
    ----------
    epoch: int
        The number of the current epoch.
    model : nn.Module
        The network to train.
    optimizer: torch.optim.Optimizer
        The optimizer to use for the training.
    train_loader : torch.utils.data.DataLoader
        A DataLoader providing access to training images and labels.   
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    """
    A simple test procedure to measure the STN performances on MNIST.
    
    Parameters
    ----------
    model : nn.Module
        The network to evaluate.
    test_loader : torch.utils.data.DataLoader
        A DataLoader providing access to test images and labels.  

    Returns
    -------
    test_loss: float
        The average test loss for the entire dataset.
    accuracy: float
        The accuracy computed on the provided dataset.
    """
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * accuracy))
        return test_loss, accuracy

def train_test(train_loader, test_loader, num_runs=5, epochs=20, use_coord=None, use_radius=False):
    """
    Train a model num_runs times on MNIST and compute the mean accuracy on
    the test set.
    
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A DataLoader providing access to training images and labels.  
    test_loader : torch.utils.data.DataLoader
        A DataLoader providing access to test images and labels.
    num_runs: int
        How many models should be trained. A higher numbers should
        provide more accurate statistics.
    epochs: int
        Number of epochs for each trained model.
    use_coord: str
        Whether the model should use CoordConv. See Net.__init__.
    use_radius: bool
        Whether CoordConv should add a third channel with radial
        information (default is False).
        Ignored if use_coord is None.

    Returns
    -------
    models: nn.Module
        The list of trained models.
    acc_list: float
        The list of accuracies for each trained model (using last epoch's weights).
    """
    models = []
    acc_list = []
    for i in range(num_runs):
        print("Starting training run {}/{} (use_coord={}, use_radius={})".format(i + 1, num_runs, use_coord, use_radius))
    
        model = Net(use_coord=use_coord, use_radius=use_radius).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Note: we would normally want to have a validation set
        # in order to select the best training hyperparameters
        # (e.g. optimizer, number of epochs, etc.)
        # but we won't bother for this toy project
        # also: my GPU would take an eternity to train the models
        for epoch in range(1, epochs + 1):
            train(epoch, model, optimizer, train_loader)
            _, accuracy = test(model, test_loader)
        acc_list.append(accuracy) # save last epoch accuracy
        models.append(model)
        
    print("Average accuracy over {} runs: {:.4f} +/- {:.4f}".format(num_runs, np.mean(acc_list), np.std(acc_list) if num_runs > 1 else 0))
    
    return models, acc_list

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
def visualize_stn(models, model_names, test_loader):
    """
    Visualize the output of the STN layer using a batch of images.
    The function will select the next batch of images from the
    provided DataLoader.
    
    Parameters
    ----------
    models : list of nn.Module
        The networks to evaluate.
    model_names: str
        Name of the models to use as graph titles.
    test_loader : torch.utils.data.DataLoader
        A DataLoader providing access to test images and labels.
    """
    # Get a batch of training data
    # use the same images for all the models
    data = next(iter(test_loader))[0].to(device)
    input_tensor = data.cpu()
    
    for model, model_name in zip(models, model_names):
        with torch.no_grad():
            transformed_input_tensor = model.stn(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Input Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images (model {})'.format(model_name))

# Reproducible DataLoader worker initialization function
# see https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate STN-based models.')
    parser.add_argument('--distort', action='store_true',
                        help='Apply random Affine transformations to MNIST images.')
    #parser.add_argument('-coord', choices=['no', 'stn', 'all'], default='no',
    #                    help="Use CoordConv instead of Conv. 'no' = no CoordConv. 'stn' = use CoordConv in the STN module only. 'all' = use CoordConv in the entire network.")
    parser.add_argument('-nruns', type=int, default=5,
                        help="Number of models to train. Average accuracy will be computed. Default 5.")
    parser.add_argument('-epochs', type=int, default=20,
                        help="Number of epochs per run. Default 20.")
    args = parser.parse_args()

    plt.ion()   # interactive mode

    # We experiment with the classic MNIST dataset. Using a standard
    # convolutional network augmented with a spatial transformer network.
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    data_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    if args.distort:
        # random affine transformation
        # we want to evaluate the network on distorted images
        # since that is where the STN shines
        random_affine = transforms.RandomAffine(degrees=45, translate=(0.2,0.2), scale=(0.85,1.15), shear=15)
        data_transforms = transforms.Compose([data_transforms, random_affine])
    
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=data_transforms),
        batch_size=64, shuffle=True, num_workers=4,
        worker_init_fn=seed_worker, generator=g)
    
    # Test dataset
    # Note that the test function always looks at the entire dataset,
    # so shuffling seed should be irrelevant for the performance evaluation.
    # We use it instead in order to extract the same sample images for
    # STN visualization.
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=data_transforms),
        batch_size=64, shuffle=True, num_workers=4,
        worker_init_fn=seed_worker, generator=g)

    # train and evaluate:
    # 1 - baseline model
    # 2 - CoordConv in the STN module
    # 3 - CoordConv with radial coordinates in the STN module
    # 4 - CoordConv in the entire network
    # 5 - CoordConv with radial coordinates in the entire network
    # NOTE: I reduced the number of epochs in order to speed up the training,
    # since my GPU is not the best
    # A higher num_runs would also be useful in order to have a more reliable
    # average.
    models = {}
    accuracies = {}
    models['baseline'],        accuracies['baseline']        = train_test(train_loader, test_loader, num_runs=args.nruns, epochs=args.epochs, use_coord=None, use_radius=False)
    models['stncoord'],        accuracies['stncoord']        = train_test(train_loader, test_loader, num_runs=args.nruns, epochs=args.epochs, use_coord='stn', use_radius=False)
    models['stncoord_radius'], accuracies['stncoord_radius'] = train_test(train_loader, test_loader, num_runs=args.nruns, epochs=args.epochs, use_coord='stn', use_radius=True)
    models['allcoord'],        accuracies['allcoord']        = train_test(train_loader, test_loader, num_runs=args.nruns, epochs=args.epochs, use_coord='all', use_radius=False)
    models['allcoord_radius'], accuracies['allcoord_radius'] = train_test(train_loader, test_loader, num_runs=args.nruns, epochs=args.epochs, use_coord='all', use_radius=True)
    
    # Visualize the STN transformation on some input batch
    # Use the first trained model for each model type
    print("Visualizing transformations.")
    visualize_stn([mlist[0] for mlist in models.values()], [mtype for mtype in models.keys()], test_loader)
    
    best_acc = 0
    best_model = None
    for model_type, acc in accuracies.items():
        mean_acc = np.mean(acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_model = model_type
        print("Average accuracy for model {}: {:.4f}".format(model_type, mean_acc))
    
    print("The '{}' model obtained the best average accuracy ({:.4f}).".format(best_model, best_acc))

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
