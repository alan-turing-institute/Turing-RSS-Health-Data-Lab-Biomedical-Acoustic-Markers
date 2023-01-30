import pickle as pkl
def read_pickle_file(file):
    '''
    Function used in 'Regression Analysis.ipynb' to read pickled dictionaries into R
    
            Parameters:
                    file: str, path to pickle file to load
    '''
    with open(file,"rb") as f:
        a = pkl.load(f)
    return a
