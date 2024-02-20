import numpy as np
import pickle

def string_to_array(s):
    # Remove square brackets and split the string
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])

# Function to save an array as a pickle
def save_array(array, filename):
    with open(filename, 'wb') as f:
        pickle.dump(array, f)

# Function to load an array from a pickle
def load_array(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)