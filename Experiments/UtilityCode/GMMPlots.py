import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
np.random.seed(seed=0)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure()
ax = Axes3D(fig)

datarange = 5
stepsize = 0.01

# Make data.
X_old = np.arange(-datarange, datarange, stepsize)
Y_old = np.arange(-datarange, datarange, stepsize)
X, Y = np.meshgrid(X_old, Y_old)

R = np.sqrt(X**2 + Y**2)
Z = R

# ##############################################
# # Set Z 
# ##############################################

number_means = 10
mean_point_set = np.random.random(size=(number_means,2))*(datarange*2)-datarange

##############################################
# Torch GMM
##############################################

gmm_means = torch.tensor(mean_point_set).to(device)
gmm_var = 1
gmm_variances = gmm_var*torch.ones_like(gmm_means).to(device)

# Create a mixture that ignores last dimension.. this should be able to handle both batched and non-batched inputs..
mixture_distribution = torch.distributions.Categorical(torch.ones(gmm_means.shape[:-1]).to(device))
component_distribution = torch.distributions.Independent(torch.distributions.Normal(gmm_means,gmm_variances),1)

GMM = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)

##############################################

grid = np.stack([X,Y]).reshape(X.shape[0],X.shape[1],2)
grid = np.stack([X,Y])
# torch_grid = torch.tensor(grid).to(device).view(X.shape[0],X.shape[1],2)
torch_grid = torch.tensor(grid.transpose(1,2,0)).to(device)
# Z = GMM.log_prob(torch_grid).detach().cpu().numpy()
Z = np.exp(GMM.log_prob(torch_grid).detach().cpu().numpy())
# Z = GMM.cdf(torch_grid).detach().cpu().numpy()

# Z = np.zeros((X.shape[0],X.shape[1]))

# print("About to eval probs")
# for i, x in enumerate(X_old):
#     for j, y in enumerate(Y_old):
#         Z[i,j] = GMM.log_prob(torch.tensor([x,y]).to(device))

print("About to plot")
# Z = gmmlogprobs
# Z = R

