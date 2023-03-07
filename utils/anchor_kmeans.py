import glob
import math
import os
import random
import shutil
import subprocess
import time
import logging
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import platform
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from utils.torch_utils import init_seeds, is_parallel
from general import kmean_anchors

if __name__ == '__main__':
    data_path = '/data/liutianchi/code/yolov5-3.0/data/fire_smog/fire_smog.yaml'
    kmeans_anchor = kmean_anchors(data_path)

