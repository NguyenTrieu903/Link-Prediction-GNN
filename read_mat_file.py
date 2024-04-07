from scipy.io import loadmat

# Load the .mat file
data = loadmat('data_SEAL/facebook.mat')
print(data)
# Access variables from the loaded data
# For example, if you have a variable named 'my_variable' in the MATLAB file
# my_variable = data['my_variable']