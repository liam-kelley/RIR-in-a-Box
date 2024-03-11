import numpy as np

def mesh2ir_rir_preprocessing(rir): #int(16384)
    '''
    mesh2ir implemented this preprocessing on the rirs of the GWA dataset.
    It was shown in an ablation study that this preprocessing improved the quality of the training.
    For sake of comparison, we include this preprocessing here.
    '''
    
    # normalize the rirs by the std, then pad with the std value
    # std_value = np.std(rir) * 10
    # std_array = np.repeat(std_value,128)
    # rir = rir/std_value
    # rir = np.concatenate([rir,std_array])

    std_value = np.std(rir) * 10
    rir = rir/std_value
    std_array = np.repeat(std_value,128)
    rir = np.concatenate([rir,std_array])

    return rir
