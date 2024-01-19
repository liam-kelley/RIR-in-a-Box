
import matplotlib.cm as cm
import numpy as np

def get_loss_experiment_cmap():
    '''Returns a colormap with an exponential scaling showing more details in the lower values'''

    # Get the original colormap
    original_cmap = cm.get_cmap('plasma')
    # Number of entries in the colormap
    n_colors = original_cmap.N
    # Generate a linear range of values between 0 and 1
    linear_values = np.linspace(0, 1, n_colors)
    # Apply an exponential scaling factor
    exponential_factor = 2  # Adjust this value as needed
    exponential_values = np.power(linear_values, exponential_factor)
    # Create a new colormap with the exponential scaling
    cmap = cm.colors.ListedColormap(original_cmap(exponential_values)).reversed()
    return cmap