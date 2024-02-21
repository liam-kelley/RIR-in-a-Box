import numpy as np

def mesh2ir_rir_preprocessing(rir, crop_length=3968): #int(16384)
    '''
    mesh2ir implemented this preprocessing on the rirs of the GWA dataset.
    It was shown in an ablation study that this preprocessing improved the quality of the training.
    For sake of comparison, we include this preprocessing here.
    '''
    length = rir.size

    # crop or pad all rirs to same length
    if(length<crop_length):
        zeros = np.zeros(crop_length-length)
        rir = np.concatenate([rir,zeros])
    else:
        rir = rir[0:crop_length]
    
    # normalize the rirs by the std, then pad with the std value
    std_value = np.std(rir) * 10
    std_array = np.repeat(std_value,128)
    rir = rir/std_value
    rir = np.concatenate([rir,std_array])

    return rir

def string_to_array(s):
    '''
    Useful for what's in that dataset csv file
    '''
    # Remove square brackets and split the string
    elements = s.strip("[]").split()
    # Convert each element to float and create a numpy array
    return np.array([float(e) for e in elements])
