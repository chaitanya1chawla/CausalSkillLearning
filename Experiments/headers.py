# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import glob, os, sys, argparse
import torch, copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import embed

import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000
import matplotlib.pyplot as plt
import tensorboardX
from scipy import stats
from absl import flags
from memory_profiler import profile as mprofile
from profilehooks import profile as tprofile
from pytorch_memlab import profile as gpu_profile
from pytorch_memlab import profile_every as gpu_profile_every
from line_profiler import LineProfiler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from IPython import embed
import pdb
import sklearn.manifold as skl_manifold
from sklearn.decomposition import PCA
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.animation import FuncAnimation
import mpl_toolkits
# import tensorflow as tf
import tempfile
import moviepy.editor as mpy
import subprocess
import h5py
import time
import unittest
import cProfile

from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial import KDTree

from sklearn.neighbors import NearestNeighbors
# from pytorch3d.loss import chamfer_distance
import random
# Removing robosuite from headers file so that we can only import it when we have a mujoco installation.
# import robosuite

import wandb
import densne

# import faulthandler; faulthandler.enable()