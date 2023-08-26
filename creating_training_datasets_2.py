from generators2 import *
from auxiliary import *
from creating_training_dataset import *
import numpy as np
from tqdm import tqdm
from random import shuffle

# Variables related to data
#       'customParam' is custom parametres of inner corelation or consistensy in single generated dataset, the lower, the greater the is mutual correlation. It should be lower than minimal number of items;
#           best values are: 1 or 'None' (then it'll be randomized).
#       'dispersion' is standard deviation of altered part's index.
#       'maxDispersion' is maximum standard devaiaton of dispersion of junk responses in dataset. For more information look at merge_q funcion in 'auxiliary.py'.
#       'junkRatio' is tuple of minimum and maximum share of "junk" responses in dataset. Should be between 0 and 1.
#       'minSize' and 'maxSize' are, respectively minimum and maximum size of generated questionnaire, whehere first number is related to number of rows (answers) and second is related to number of items.

# For some functions
#        'noise' - is another custom parametres related to certain generetors, look for corresponding function to see more details

# Variables related to process of creating
#       'numberOfJunkQuestionaries' - number of 'junk' questionaries (of each type in MixedGenerator), should be a natural number.
#       'trueQuestionariesPerJunk' - number of 'correct' questionaries per each junk one.
#       'bootstrap' (false by default) - generating bootstraped questionaries instead of control_group.
#       'padding'   (true by default) - resize each questionnaire to max size by adding zeroes.
#       'returnParametres' - when 'True' each questionaries have aditional vector with ex ante and ex post parametres of 'junkiness'
 
 
# MixedGenerator is basic function for creating dataset of junk and correct datasets

def MixedGenerator(numberOfJunkQuestionaries = 1, customParam = 1, junkRatio = (0.2,0.4), maxDispersion = 0.25, returnParametres = False, truePerJunkRatio = 1, minSize = (20,5), maxSize = (500,40), bootstrap = False, padding = True, noise1=0.01, noise2=0.01):
    gen = generator(numberOfJunkQuestionaries, returnParametres, truePerJunkRatio, minSize, maxSize, bootstrap, padding)
    x1 =         gen.ufo_junk_group(customParam, junkRatio, maxDispersion)
    x2 =   gen.corelated_junk_group(customParam, junkRatio, maxDispersion, noise1)
    x3 = gen.uncorelated_junk_group(customParam, junkRatio, maxDispersion)
    x4 =       gen.equal_junk_group(customParam, junkRatio, maxDispersion, noise2)
    x_control = generator(numberOfJunkQuestionaries, returnParametres, truePerJunkRatio, minSize, maxSize, bootstrap, padding)
    datasets = x1+x2+x3+x4
    shuffle(datasets)

    return(datasets)

# generator class
class generator:
    def __init__(self, numberOfJunkQuestionaries, returnParametres, truePerJunkRatio = 1, minSize = (20,5), maxSize = (500,40), bootstrap = False, padding = False):
        self.njq = numberOfJunkQuestionaries
        self.tpj = truePerJunkRatio
        self.ntq = numberOfJunkQuestionaries * truePerJunkRatio
        self.maxRows = maxSize[0]
        self.maxItems = maxSize[1]
        self.minRows = minSize[0]
        self.minItems = minSize[1]
        self.bootstrap = bootstrap
        self.padding = padding
        self.returnParametres = returnParametres
        
   
    
        
    def control_group(self, inputParam = None):
        datasets = []
        for _ in tqdm(range(self.ntq)):
        
            # Set random parametres
            nRows  = np.random.randint(self.minRows, self.maxRows)
            nItems = np.random.randint(self.minItems, self.maxItems)
            
            if inputParam == None:
                param = np.random.randint(1, nItems)
            else:
                param = inputParam
            
            # Create dataset
            q = control(nRows, nItems, param)
            
            # Normalize dataset
            q = normalize(q)
            
            if self.returnParametres == True:
                parametres = ('control', param)
                datasets.append((q, 0, parametres))
            else:
                datasets.append((q, 0))
        
        if self.padding == True:
            datasets = padding_dataset(datasets, self.maxRows, self.maxItems)
            
        return(datasets)

    def ufo_junk_group(self, customParam = None, junkRatio = (0.40, 0.60), maxDispersion = 0.25):
        datasets = []
        self.ntq
        for _ in tqdm(range(self.njq)):
                
            # Set random parametres for questionaries
            nRows  = np.random.randint(self.minRows, self.maxRows)
            nItems = np.random.randint(self.minItems, self.maxItems)
            
            # Set number of rows in 1st part
            n1Rows = np.random.randint(junkRatio[0] * nRows, junkRatio[1] * nRows)
            n2Rows = nRows - n1Rows
            
            if customParam == None:
                param = np.random.randint(1, nItems)
            else:
                param = customParam
            
            # Create two parts of dataset (junk_ufo() is the same function as control; see generators2)
            q1 = control(n1Rows, nItems, param)
            q2 = junk_ufo(n2Rows, nItems, param)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1Rows)
            dispersion = np.random.random() * n1Rows * maxDispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            # Normalize dataset
            q = normalize(q)            
            
            if self.returnParametres == True:
                parametres = ('ufo_junk', param, n2Rows/nRows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))   
        
       # Add bootstrap if bootstrap is on     
        if self.bootstrap == True:
            datasets = datasets + bootstrap(datasets, self.tpj, self.returnParametres)
            
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.maxRows, self.maxItems)  
        return(datasets)

    def uncorelated_junk_group(self, customParam = None, junkRatio = (0.40, 0.60), maxDispersion = 0.25):
        datasets = []
        for _ in tqdm(range(self.njq)):
                
            # Set random parametres for questionaries
            nRows  = np.random.randint(self.minRows, self.maxRows)
            nItems = np.random.randint(self.minItems, self.maxItems)
            
            # Set number of rows in 1st part
            n1Rows = np.random.randint(junkRatio[0] * nRows, junkRatio[1] * nRows)
            n2Rows = nRows - n1Rows
            
            if customParam == None:
                param = np.random.randint(1, nItems)
            else:
                param = customParam
            
            # Create two parts of dataset
            q1 = control(n1Rows, nItems, param)
            q2 = uncorelated_junk(n2Rows, nItems)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1Rows)
            dispersion = np.random.random() * n1Rows * maxDispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            # Normalize dataset
            q = normalize(q)
                        
            if self.returnParametres == True:
                parametres = ('uncorelated_junk', param, n2Rows/nRows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))
            
       # Add bootstrap if bootstrap is on     
        if self.bootstrap == True:
            datasets = datasets + bootstrap(datasets, self.tpj, self.returnParametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.maxRows, self.maxItems)
        return(datasets)

    # 'noise' is custom parametre of noise in correlation. If noise = 0 corelation are perfect, if noise >1 (eq.=1) correlations are not "ideal"
    def corelated_junk_group(self, customParam = None, junkRatio = (0.40, 0.60), maxDispersion = 0.25, noise = 0):
        datasets = []
        for _ in tqdm(range(self.njq)):                
            # Set random parametres for questionaries
            nRows  = np.random.randint(self.minRows, self.maxRows)
            nItems = np.random.randint(self.minItems, self.maxItems)
            
            # Set number of rows in 1st part
            n1Rows = np.random.randint(junkRatio[0] * nRows, junkRatio[1] * nRows)
            n2Rows = nRows - n1Rows
            
            if customParam == None:
                param = np.random.randint(1, nItems)
            else:
                param = customParam
            
            # Create two parts of dataset
            q1 = control(n1Rows, nItems, param)
            q2 = corelated_junk(n2Rows, nItems, noise)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1Rows)
            dispersion = np.random.random() * n1Rows * maxDispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            # Normalize dataset
            q = normalize(q)   
            
            if self.returnParametres == True:
                parametres = ('corelated_junk_group', param, n2Rows/nRows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))   
        
       # Add bootstrap if bootstrap is on     
        if self.bootstrap == True:
            datasets = datasets + bootstrap(datasets, self.tpj, self.returnParametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.maxRows, self.maxItems)
        return(datasets)

    # 'noise' is custom parametre of noise in correlation. If noise = 0 all junk responses are the same, if bigger, there is some noise
    def equal_junk_group(self, customParam = None, junkRatio = (0.40, 0.60), maxDispersion = 0.25, noise = 0):
        datasets = []
        for _ in tqdm(range(self.njq)):  
            # Set random parametres for questionaries
            nRows  = np.random.randint(self.minRows, self.maxRows)
            nItems = np.random.randint(self.minItems, self.maxItems)
            
            # Set number of rows in 1st part
            n1Rows = np.random.randint(junkRatio[0] * nRows, junkRatio[1] * nRows)
            n2Rows = nRows - n1Rows
            
            if customParam == None:
                param = np.random.randint(1, nItems)
            else:
                param = customParam
            
            # Create two parts of dataset
            q1 = control(n1Rows, nItems, param)
            q2 = corelated_junk(n2Rows, nItems, noise)   
            
            # Set parametres of junk order 
            location = np.random.randint(0,n1Rows)
            dispersion = np.random.random() * n1Rows * maxDispersion
            
            # Merge
            q=merge_q(q1,q2,location, dispersion)
            
            # Normalize dataset
            q = normalize(q)
            
            if self.returnParametres == True:
                parametres = ('equal_junk_group', param, n2Rows/nRows, location, dispersion)
                datasets.append((q, 1, parametres))
            else:
                datasets.append((q, 1))                
       
       # Add bootstrap if bootstrap is on     
        if self.bootstrap == True:
            datasets = datasets + bootstrap(datasets, self.tpj, self.returnParametres)
        
        shuffle(datasets)
        if self.padding == True:
            datasets = padding_dataset(datasets, self.maxRows, self.maxItems)
        return(datasets)