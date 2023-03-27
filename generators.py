import numpy as np
rng = np.random.default_rng()
import pandas as pd

def control_group(max_rows = 500, max_items = 40):
    # Random number of responses
    n_true=np.random.randint(0, max_rows, size=1)

    # Random number of items
    n=np.random.randint(1, max_items, size=1)
    n = int(n)

    # Random means
    means = 2*np.random.random(size=n)-1

    # Random covariance matrix
    b = 2*np.random.random(size=(n,n))-1
    b_symm = (b + b.T)/2

    # Matrix of responses
    responses = rng.multivariate_normal(means, b_symm, check_valid="ignore", size=n_true, method='eigh')
    responses = pd.DataFrame(responses)
    
    return(responses)

def random_junk_group(max_rows = 500, max_items = 40):
    # Random number of responses
    n_resp=np.random.randint(20, max_rows, size=1)
    n_true=np.random.randint(n_resp/3, 2*n_resp/3, size=1)
    n_false = n_resp - n_true

    # Random number of items
    n=np.random.randint(1, max_items, size=1)
    n = int(n)

    # Random means
    means = 2*np.random.random(size=n)-1

    # Random covariance matrix
    b = 2*np.random.random(size=(n,n))-1
    b_symm = (b + b.T)/2

    # Matrix of responses
    responses = rng.multivariate_normal(means, b_symm, check_valid="ignore", size=n_true, method='eigh')
    responses = pd.DataFrame(responses)

    # Add Order
    responses.insert(0, 'Order', range(1, 1+len(responses)))

    # Junk
    # Random responses
    b_random = np.eye(n)
    junk_means = 2*np.random.random(size=n)-1
    junk_random = rng.multivariate_normal(junk_means, b_random, check_valid="ignore", size=n_false)
    junk_random = pd.DataFrame(junk_random)

    junk_random.insert(0, 'Order', np.random.normal(
                                                    loc= ((np.random.random()*5*n_true)/6 + 1/6),
                                                    scale=np.random.random()*n_true/6,
                                                    size=n_false
                                                    ))
    
    full = pd.concat([responses, junk_random])
    full = full.sort_values(by=['Order'])
    del full['Order']

    return(full)

def flat_junk_group(max_rows = 500, max_items = 40):
    # Random number of responsen
    n_resp=np.random.randint(20, max_rows, size=1)
    n_true=np.random.randint(n_resp/3, 2*n_resp/3, size=1)
    n_false = n_resp - n_true

    # Random number of items
    n=np.random.randint(1, max_items, size=1)
    n = int(n)

    # Random means
    means = 2*np.random.random(size=n)-1

    # Random covariance matrix
    b = 2*np.random.random(size=(n,n))-1
    b_symm = (b + b.T)/2

    # Matrix of responses
    responses = rng.multivariate_normal(means, b_symm, check_valid="ignore", size=n_true, method='eigh')
    responses = pd.DataFrame(responses)

    # Add Order
    responses.insert(0, 'Order', range(1, 1+len(responses)))

    b_flat = np.ones([n, n])
    junk_means = 2*np.random.random(size=n)-1
    junk_flat = rng.multivariate_normal(junk_means, b_flat, check_valid="ignore", size=n_false)
    junk_flat = pd.DataFrame(junk_flat)
    junk_flat.insert(0, 'Order', np.random.normal(
                                                    loc= ((np.random.random()*5*n_true)/6 + 1/6),
                                                    scale=np.random.random()*n_true/6,
                                                    size=n_false
                                                    ))
    
    full = pd.concat([responses, junk_flat])
    full = full.sort_values(by=['Order'])
    del full['Order']
    
    return(full)
    
def ufo_junk_group(max_rows = 500, max_items = 40):
    # Random number of responses
    n_resp=np.random.randint(20, max_rows, size=1)
    n_true=np.random.randint(n_resp/3, 2*n_resp/3, size=1)
    n_false = n_resp - n_true

    # Random number of items
    n=np.random.randint(1, max_items, size=1)
    n = int(n)


    # Random means
    means = 2*np.random.random(size=n)-1

    # Random covariance matrix
    b = 2*np.random.random(size=(n,n))-1
    b_symm = (b + b.T)/2

    # Matrix of responses
    responses = rng.multivariate_normal(means, b_symm, check_valid="ignore", size=n_true, method='eigh')
    responses = pd.DataFrame(responses)

    # Add Order
    responses.insert(0, 'Order', range(1, 1+len(responses)))

    # Junk Ufo
    b_ufo = 2*np.random.random(size=(n,n))-1
    b_symm_ufo = (b_ufo + b_ufo.T)/2
    means_ufo = 2*np.random.random(size=n)-1
    junk_ufo = rng.multivariate_normal(means_ufo, b_symm_ufo, check_valid="ignore", size=n_false, method='eigh')
    junk_ufo = pd.DataFrame(junk_ufo)
    junk_ufo.insert(0, 'Order', np.random.normal(
                                                    loc= ((np.random.random()*5*n_true)/6 + 1/6),
                                                    scale=np.random.random()*n_true/4,
                                                    size=n_false
                                                    ))

    full = pd.concat([junk_ufo, responses])
    full = full.sort_values(by=['Order'])
    del full['Order']
    return(full) 