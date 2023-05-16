from generators2 import *
from auxiliary import *
from creating_training_dataset import *
import numpy as np
from tqdm import tqdm
from random import shuffle


def MixedGenerator(numberOfJunkQuestionaries = 1, input_param = 1, junk_ratio = (0.2,0.4), max_dispersion = 0.25, returnParametres = False, trueQuestionariesPerJunk = 1, min_size = (20,5), max_size = (500,40), bootstrap = False, padding = False, noise1=0.01, noise2=0.01):
    x1= generator(numberOfJunkQuestionaries, returnParametres, trueQuestionariesPerJunk, min_size, max_size, bootstrap).ufo_junk_group(input_param, junk_ratio, max_dispersion)
    x2= generator(numberOfJunkQuestionaries, returnParametres, trueQuestionariesPerJunk, min_size, max_size, bootstrap).corelated_junk_group(input_param, junk_ratio, max_dispersion, noise1)
    x3= generator(numberOfJunkQuestionaries, returnParametres, trueQuestionariesPerJunk, min_size, max_size, bootstrap).uncorelated_junk_group(input_param, junk_ratio, max_dispersion)
    x4= generator(numberOfJunkQuestionaries, returnParametres, trueQuestionariesPerJunk, min_size, max_size, bootstrap).equal_junk_group(input_param, junk_ratio, max_dispersion, noise2)   
    datasets = x1+x2+x3+x4
    shuffle(datasets)
    
    if padding == True:
        datasets = padding_dataset(datasets)
    return(datasets)

# generator class
class generator:
    def __init__(self, numberOfJunkQuestionaries = 1, return_parametres = False, truePerJunkRatio = 1, minSize = (20,5), maxSize = (500,40), bootstrap = False, padding = False):
        self.njq = numberOfJunkQuestionaries
        self.tpj = truePerJunkRatio
        self.ntq = numberOfJunkQuestionaries * truePerJunkRatio
        self.max_rows = maxSize[0]
        self.max_items = maxSize[1]
        self.min_rows = minSize[0]
        self.min_items = minSize[1]
        self.bootstrap = bootstrap
        self.padding = padding
        self.return_parametres = return_parametres
        
    # Variables for all methods
    #       'input_param' is custom parametres of inbetween corelation, the lower, the greater the is mutual correlation, it should be lower than minimal number of items;
    #           best values are: 0 or 'None' (then it'll be randomized).
    #       'dispersion' is standard deviation of altered part's index.
    #       'junk_ratio' is tuple of minimum and maximum share of "junk" responses in dataset. Should be between 0 and 1.
    #       'max_dispersion' is maximum standard devaiaton of dispersion of junk responses in dataset. For more information look at merge_q funcion in auxiliary.py
    
    
        
    def control_group(self, inputParam = None):
        datasets = []
        for _ in tqdm(range(self.ntq)):
        
            # Set random parametres
            n_rows  = np.random.randint(self.min_rows, self.max_rows)
            n_items = np.random.randint(self.min_items, self.max_items)
            
            if inputParam == None:
                param = np.random.randint(1, n_items)
            else:
                param = inputParam
            
            # Create dataset
            q = control(n_rows, n_items, param)
            
            if self.return_parametres == True:
                parametres = ('control',param)
                datasets.append((q, 0, parametres))
            else:
                datasets.append((q, 0))
        
        if self.padding == True:
            datasets = padding_dataset(datasets, self.max_rows, self.max_items)    
        return(datasets)

    def ufo_junk_group(self, input_param = None, junk_ratio = (0.40,0.60), max_dispersion = 0.25):
        datasets = []
        self.ntq
        for _ in tqdm(range(self.njq)):
                
            # Set random parametres for questionaries
            n_rows  = np.random.randint(self.min_rows, self.max_rows)
            n_items = np.random.randint(self.min_items, self.max_items)
            
            # Set number of rows in 1st part
            n1_rows = np.random.randint(junk_ratio[0] * n_rows, junk_ratio[1] * n_rows)
            n2_rows = n_rows - n1_rows
            
            if input_param == None:
                param = np.random.randint(1, n_items)
            else:
                param = input_param
            
            # Create two parts of dataset (junk_ufo() is the same function as control; see generators2)
            q1 = control(n1_rows, n_items, param)
            q2 = junk_ufo(n2_rows, n_items, param)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1_rows)
            dispersion = np.random.random() * n1_rows * max_dispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            if self.return_parametres == True:
                parametres = ('ufo_junk', param, n2_rows/n_rows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))   
        
       # Add control if bootstrap is off     
        if self.bootstrap == False:
            datasets = datasets + generator(self.ntq, self.return_parametres).control_group(param)
        else:
            datasets = datasets + bootstrap(datasets, self.tpj, self.return_parametres)
            
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.max_rows, self.max_items)  
        return(datasets)

    def uncorelated_junk_group(self, input_param = None, junk_ratio = (0.40,0.60), max_dispersion = 0.25):
        datasets = []
        for _ in tqdm(range(self.njq)):
                
            # Set random parametres for questionaries
            n_rows  = np.random.randint(self.min_rows, self.max_rows)
            n_items = np.random.randint(self.min_items, self.max_items)
            
            # Set number of rows in 1st part
            n1_rows = np.random.randint(junk_ratio[0] * n_rows, junk_ratio[1] * n_rows)
            n2_rows = n_rows - n1_rows
            
            if input_param == None:
                param = np.random.randint(1, n_items)
            else:
                param = input_param
            
            # Create two parts of dataset
            q1 = control(n1_rows, n_items, param)
            q2 = uncorelated_junk(n2_rows, n_items)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1_rows)
            dispersion = np.random.random() * n1_rows * max_dispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            if self.return_parametres == True:
                parametres = ('uncorelated_junk', param, n2_rows/n_rows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))
            
       # Add control if bootstrap is off     
        if self.bootstrap == False:
            datasets = datasets + generator(self.ntq, self.return_parametres).control_group(param)
        else:
            datasets = datasets + bootstrap(datasets, self.tpj, self.return_parametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.max_rows, self.max_items)
        return(datasets)

    # 'noise' is custom parametre of noise in correlation. If noise = 0 corelation are perfect, if noise >1 (eq.=1) correlations are not "ideal"
    def corelated_junk_group(self, input_param = None, junk_ratio = (0.40,0.60), max_dispersion = 0.25, noise = 0):
        datasets = []
        for _ in tqdm(range(self.njq)):                
            # Set random parametres for questionaries
            n_rows  = np.random.randint(self.min_rows, self.max_rows)
            n_items = np.random.randint(self.min_items, self.max_items)
            
            # Set number of rows in 1st part
            n1_rows = np.random.randint(junk_ratio[0] * n_rows, junk_ratio[1] * n_rows)
            n2_rows = n_rows - n1_rows
            
            if input_param == None:
                param = np.random.randint(1, n_items)
            else:
                param = input_param
            
            # Create two parts of dataset
            q1 = control(n1_rows, n_items, param)
            q2 = corelated_junk(n2_rows, n_items, noise)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1_rows)
            dispersion = np.random.random() * n1_rows * max_dispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            if self.return_parametres == True:
                parametres = ('corelated_junk_group', param, n2_rows/n_rows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))   
        
       # Add control if bootstrap is off     
        if self.bootstrap == False:
            datasets = datasets + generator(self.ntq, self.return_parametres).control_group(param)
        else:
            datasets = datasets + bootstrap(datasets, self.tpj, self.return_parametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.max_rows, self.max_items)
        return(datasets)

    # 'noise' is custom parametre of noise in correlation. If noise = 0 all junk responses are the same, if bigger, there is some noise
    def equal_junk_group(self, input_param = None, junk_ratio = (0.40,0.60), max_dispersion = 0.25, noise = 0):
        datasets = []
        for _ in tqdm(range(self.njq)):  
            # Set random parametres for questionaries
            n_rows  = np.random.randint(self.min_rows, self.max_rows)
            n_items = np.random.randint(self.min_items, self.max_items)
            
            # Set number of rows in 1st part
            n1_rows = np.random.randint(junk_ratio[0] * n_rows, junk_ratio[1] * n_rows)
            n2_rows = n_rows - n1_rows
            
            if input_param == None:
                param = np.random.randint(1, n_items)
            else:
                param = input_param
            
            # Create two parts of dataset
            q1 = control(n1_rows, n_items, param)
            q2 = corelated_junk(n2_rows, n_items, noise)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1_rows)
            dispersion = np.random.random() * n1_rows * max_dispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            if self.return_parametres == True:
                parametres = ('equal_junk_group', param, n2_rows/n_rows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))                
       
       # Add control if bootstrap is off     
        if self.bootstrap == False:
            datasets = datasets + generator(self.ntq, self.return_parametres).control_group(param)
        else:
            datasets = datasets + bootstrap(datasets, self.tpj, self.return_parametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.max_rows, self.max_items)
        return(datasets)