import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--n_batches", type=int, default=50, help="number of batches")
parser.add_argument("--filename_Losses", type=str, default='Loss_model.txt', help="filename of losses")
opt = parser.parse_args()

path = r'./'
num = opt.n_epochs * opt.n_batches
with open(os.path.join(path, opt.filename_Losses)) as f:
    next(f)
    Losses = f.readlines()[0:num]
    DLoss = []
    GLoss = []
    for Loss in Losses:
        DLoss.append(Loss.strip('\n').split(' ')[6].strip(']'))
        GLoss.append(Loss.strip('\n').split(' ')[9].strip(']'))
x = np.arange(0, num)
DLoss = np.array(DLoss, dtype=float)
GLoss = np.array(GLoss, dtype=float)
plt.plot(x, DLoss, label='Discriminator Loss', linewidth=1)
plt.plot(x, GLoss, label='Generator Loss', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curves of AnimalGAN {}'.format(opt.filename_Losses.split('_')[1].strip('.txt')))
plt.legend()

if not os.path.exists(os.path.join(path, 'LossCurves/')):
    os.makedirs(os.path.join(path, 'LossCurves/'))
