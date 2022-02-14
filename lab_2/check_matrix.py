import matplotlib.pyplot as plt

filename = "matrix.txt"

with open(filename) as f:
    matrix = [list(map(float, row.split())) for row in f.readlines()]

    plt.imshow(matrix)
    plt.show()
