from generators import *
from preprocessing import *
import numpy as np
import pandas as pd
from tqdm import tqdm

# flawed an outdated
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

# simple and slow
def create_dataset_basic(n):
    datasets = padding(control_group())
    goals = np.concatenate([np.ones((3*n,1)), np.zeros((3*n,1))])
    
    for i in tqdm(range(3*n-1)):
        input = control_group()
        datasets = np.concatenate([datasets, padding(input)])
        
    for i in tqdm(range(n)):
        input = random_junk_group()
        datasets = np.concatenate([datasets, padding(input)])
        
    for i in tqdm(range(n)):
        input = flat_junk_group()
        datasets = np.concatenate([datasets, padding(input)])

    for i in tqdm(range(n)):
        input = ufo_junk_group()
        datasets = np.concatenate([datasets, padding(input)])

    return datasets, goals


# trivial and slow
def create_dataset_trivial(n):
    datasets = padding(control_group())
    goals = np.concatenate([np.ones((n,1)), np.zeros((n,1))])
    
    for i in tqdm(range(n-1)):
        input = control_group()
        datasets = np.concatenate([datasets, padding(input)])
        
    for i in tqdm(range(n)):
        input = junk_test()
        datasets = np.concatenate([datasets, padding(input)])

    return datasets, goals
