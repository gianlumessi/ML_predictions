import numpy as np
from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV
import pandas as pd

print(np.linspace(5, 60, num=56))

import os
dir_path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, '/')
fn = dir_path + '/Data/btc-usd-coingecko_2015.csv'

df = pd.read_csv(fn)
print(df.head())

