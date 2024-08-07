# import pickle
#
# # Path to your .pkl file
# file_path = './exp/20240708-135613/EEGModel-efficientnet_Optim-adam_LR-0.0005_Epochs-40/args.pkl'
#
# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#
# # Now `data` contains the deserialized Python object
# print(data)
import numpy as np

# Define your matrix
A = np.array([[3, -2, 4],
              [-2, 6, 2],
              [4, 2, 3]])

# Calculate eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

