import pandas as pd
import numpy as np
import scipy
import patsy
import pickle
import scipy.sparse
import matplotlib.pyplot as plt

from statistics import median
from scipy.stats import gaussian_kde
from statsmodels.formula.api import ols
from tqdm import tqdm

#using barra data directories, build a backtest
barra_dir = '../../path/to/file'

data = {}
