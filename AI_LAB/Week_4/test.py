import numpy as np

# Relation matrix for nodes A, B, C
R = np.array([
    [1, 1, 0],  # A
    [1, 1, 1],  # B
    [0, 1, 1]   # C
])

def is_reflexive(matrix):
    return all(matrix[i][i] == 1 for i in range(len(matrix)))

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

print("Reflexive:", is_reflexive(R))    # Checks A->A, B->B, C->C
print("Symmetric:", is_symmetric(R))    # Checks R[i][j] == R[j][i]
