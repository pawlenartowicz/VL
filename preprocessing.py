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

    # for i in range(ad_range - 1):
    #     dataset = pd.concat([dataset,
    #                          padding(
    #                              input.sample(frac=1).reset_index()
    #                          )])
    #
    # dataset = pd.concat([dataset, padding(input)])


    preprocessed_dataset = (dataset.to_numpy(), goal)
    return(preprocessed_dataset)