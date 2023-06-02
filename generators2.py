import numpy as np
rng = np.random.default_rng()
import pandas as pd

# control_gropu to generate 'valid' questionaries
def control(n_rows, n_items, param):
    # Random means rescaled to -1:1
    means = 2*np.random.random(size=n_items)-1

    # Random covariance matrix
    b = np.random.normal(size=(n_items,param))
    cov_matrix = np.matmul(b, b.T) + np.diagflat(np.random.random(size=n_items))
    
    # Matrix of responses
    responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, method='cholesky')
    
    return(responses)

# junk_ufo_group is exacly the same as 'control', added for better understanding the code
def junk_ufo(n_rows, n_items, param=1):
    # Random means rescaled to -1:1
    means = 2*np.random.random(size=n_items)-1

    # Random corelated covariance matrix
    b = np.random.normal(size=(n_items,param))
    cov_matrix = np.matmul(b, b.T) + np.diagflat(np.random.random(size=n_items))
    
    # Matrix of responses
    responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, method='cholesky')
    
    return(responses)

# group of random, uncorelated responses
def uncorelated_junk(n_rows, n_items):
    # Random means rescaled to -1:1
    means = 2*np.random.random(size=n_items)-1  

    cov_matrix = np.diagflat(np.random.random(size=n_items))

    responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, method='cholesky')  
    return(responses)


# group of random, highly corelated responses
# @noise should be in [0, 1]
def corelated_junk(n_rows, n_items, noise):
    # Random means rescaled to -1:1
    means = 2*np.random.random(size=n_items)-1  
    
    if noise == 0:
        cov_matrix = np.ones([n_items, n_items])

        responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, check_valid='ignore', method='svd')  
        return(responses)        
    else:
        cov_matrix = np.ones([n_items, n_items]) + np.diagflat(np.random.random(size=n_items))*noise

        responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, method='cholesky')  
        return(responses)
    
# group of random, uniform responses
# @noise should be in [0, 1]
def equal_junk(n_rows, n_items, noise = 0):
    # Random means rescaled to -1:1
    means = 2*np.random.random(size=n_items)-1  
    
    if noise == 0:
        cov_matrix = np.zeros([n_items, n_items])

        responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, check_valid='ignore', method='svd')  
        return(responses)        
    else:
        cov_matrix = np.zeros([n_items, n_items]) + np.eye(n_items)*noise
        
        responses = rng.multivariate_normal(means, cov_matrix, size=n_rows, method='cholesky')  
        return(responses)