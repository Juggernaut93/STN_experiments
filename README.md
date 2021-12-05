# STN_experiments
This repository containes some toy experiments on a simple Spatial Transformer Network (STN) CNN.

Code is partially based on [this PyTorch tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html) by Ghassen Hamrouni.

Code was tested on Python 3.7, PyTorch 1.10, CUDA 10.2.

## Experimental setting
6 different model configurations were trained and tested on a modified MNIST dataset. Random affine transformation were applied to the images in order to evaluate the potential of the STN.

Due to hardware constraints, each model was trained for a fixed number of epochs (20). Each model was trained 5 times in order to compute the average classification accuracy and partially compensating randomness.

Note that the script has in place partial measures to ensure the repeatability of the experiments. Note, however, that some of the PyTorch operations employed do not have a deterministic version, so results may vary.

## Models and results
The 6 trained models are the following:
1. *Baseline*: the original simple STN-based CNN from the cited PyTorch tutorial.
2. *CoordConv_STN*: Conv layers in the STN module were replaced by CoordConv layers ([https://arxiv.org/abs/1807.03247](paper)). [https://github.com/mkocabas/CoordConv-pytorch](This implementation) was used.
3. *CoordConv_R_STN*: same as #2, but the additional channel with radial coordinates was added. Note that I fixed the radial channel computation by correctly re-centering the origin.
4. *CoordConv_all*: Conv layers were replaced by CoordConv layers in the entire network.
5. *CoordConv_R_all*: same as #4, with radial coordinates enabled.
6. *CoordConv_R_all_thetaprop*: same as #5, but I expanded the concept of CoordConv by adding 6 additional channels, each representing the value of the 6 affine transformation parameters computed by the STN module. The additional information was provided to the 2 main convolutional layers and the first FC layer of the network.

| Model | Accuracy over 5 runs (mean ± std) |
| -------------------------- | ------------- |
| Baseline                   | 0.9142 ± 0.0762* |
| CoordConv_STN              | 0.9548 ± 0.0116 |
| CoordConv_R_STN            | 0.9494 ± 0.0148 |
| CoordConv_all              | 0.9563 ± 0.0026 |
| CoordConv_R_all            | 0.9413 ± 0.0154 |
| CoordConv_R_all_thetaprop  | 0.9562 ± 0.0053 |

\*Since no validation set was employed for hyperparameter selection (training duration, learning rate, optimizer, etc.), the accuracy of a model was just selected as the accuracy obtained after the 20th epoch of training. One of the 5 runs for the baseline model happened to have a sudden decay in accuracy from ~95% to ~76% at exactly the 20th epoch. Realistically, the lower accuracy of the baseline model is due to a statistical fluke, since the other models also experienced some accuracy dips during training, but were lucky enough not to have them happen on the 20th epoch.

Results are comparable for all the 6 models, although a longer (and more refined) training procedure would be necessary in order to draw reliable conclusions.

## Script usage
`python spatial_transformer_experiments.py [--distort] [-nruns 5] [-epochs 20]`

Trains and evaluates the 6 models on MNIST.

`--distort` tells the script to use the distorted MNIST images.

`-nruns N` specifies how many times each model should be trained for metrics averaging.

`-epochs K` specifies how many epoch each model should trained for.
