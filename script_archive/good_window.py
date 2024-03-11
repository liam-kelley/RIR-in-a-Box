
def get_a_good_window_length(fs=16000, center_frequency=500):
    '''
    This funciton returns a "good" window length for a specific filter center_frequency.
    Since having a changing window length was confusing, it's now deprecated
    '''
    filter_transition_band = center_frequency *1 #* 0.2 # a good approximation.
    filter_order = fs / filter_transition_band # a good approximation.
    window_length = (filter_order // 2)*2  + 1
    return window_length