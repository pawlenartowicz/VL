from generators import *
import numpy as np
import pandas as pd
from tqdm import tqdm

def padding_array(input, max_rows, max_items):
    n_rows = input.shape[0]
    n_items = input.shape[1]
    output = np.pad(input, pad_width=((0, max_rows - int(n_rows)),(0,max_items - n_items)))
    return(output) 

def padding_dataset(input, max_rows, max_items):
    output = []
    for i in tqdm(range(len(input))):
        output.append([])
        output[i] = list(input[i])
        output[i][0] = padding_array(input[i][0], max_rows, max_items)
        output[i] = tuple(output[i])
    return(output)

def merge_q(q1,q2,location, dispersion):
    # Add default order to q1
    q1 = np.c_[range(len(q1)),q1]
    
    # Add junk order to q2
            
    order = np.random.normal(
                            loc   = location,
                            scale = dispersion,
                            size  = len(q2)
                            )
    q2 = np.c_[order,q2]            
            
    # Merge, sort, and delete order
    q = np.r_[q1,q2]
    q = q[q[:,0].argsort()]
    q = np.delete(q, 0, 1)
    
    return(q)

def VL_shuffle(x):
    x = np.c_[np.random.permutation(len(x)), x]
            
    # Sort and delete order
    x = x[x[:,0].argsort()]
    x = np.delete(x, 0, 1)
    return(x)

def bootstrap(in_datasets, tpj, return_parametres):
    out_datasets = []
    njq = len(in_datasets)
    for i in tqdm(range(njq)):
        for _ in range(tpj):
            
            x = VL_shuffle(in_datasets[i][0])
            
            if return_parametres == True:
                out_datasets.append( (x, 0, in_datasets[i][2]) ) 
            else:
                out_datasets.append( (x, 0) )
    return(out_datasets)

# deprecated
def VL_preprocessing(input):
    ad_range = 2
    m=np.random.randint(0, ad_range, size=1)
    goal = np.zeros((ad_range,1))
    goal[m] = 1
    dataset = pd.DataFrame(columns=range(40))

    for i in range(ad_range):
        if(i==m):
            dataset = pd.concat([dataset, padding(input)])
        else:
            dataset = pd.concat([dataset, 
                       padding(
                               input.sample(frac=1).reset_index()
                               )])
    preprocessed_dataset = (dataset.to_numpy(), goal)
    return(preprocessed_dataset)

# deprecated
def padding(full, max_rows = 500, max_items = 40):
    n_resp = full.shape[0]
    n = full.shape[1]
    full = np.pad(full, pad_width=((0, max_rows - int(n_resp)),(0,max_items - n)))
    full = pd.DataFrame(full)
    return(full) 