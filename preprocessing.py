from generators import *
import numpy as np
import pandas as pd

def padding(full, max_rows = 500, max_items = 40):
    n_resp = full.shape[0]
    n = full.shape[1]
    full = np.pad(full, pad_width=((0, max_rows - int(n_resp)),(0,max_items - n)))
    full = pd.DataFrame(full)
    return(full) 

def VL_preprocessing(input):
    m=np.random.randint(0, 100, size=1)
    goal = np.zeros((100,1))
    goal[m] = 1
    dataset = pd.DataFrame(columns=range(40))

    for i in range(100):
        if(i==m):
            dataset = pd.concat([dataset, padding(input)])
        else:
            dataset = pd.concat([dataset, 
                       padding(
                               input.sample(frac=1).reset_index()
                               )])
    
    preprocessed_dataset = (dataset.to_numpy(), goal)
    return(preprocessed_dataset)
