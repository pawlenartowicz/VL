from generators import *
from preprocessing import *
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_dataset(n):
    datasets = VL_preprocessing(random_junk_group())[0]
    goals = VL_preprocessing(random_junk_group())[1]



    for i in tqdm(range(n-1)):
        input = random_junk_group()
        datasets = np.concatenate([datasets, VL_preprocessing(input)[0]])
        goals = np.concatenate([goals, VL_preprocessing(input)[1]])

    for i in tqdm(range(n)):
        input = flat_junk_group()
        datasets = np.concatenate([datasets, VL_preprocessing(input)[0]])
        goals = np.concatenate([goals, VL_preprocessing(input)[1]])

    for i in tqdm(range(n)):
        input = ufo_junk_group()
        datasets = np.concatenate([datasets, VL_preprocessing(input)[0]])
        goals = np.concatenate([goals, VL_preprocessing(input)[1]])

    return datasets, goals
