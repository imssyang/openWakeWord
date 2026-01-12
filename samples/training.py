import os
import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import openwakeword
import scipy
import torch
from tqdm import tqdm


squad_dataset = datasets.load_dataset('rajpurkar/squad')
print(squad_dataset['train'][0])

