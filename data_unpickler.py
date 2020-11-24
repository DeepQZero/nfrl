import numpy as np

data = np.load('week1.npy')
positions = []
for i in range(10000):
    if data[i][16] == '75' and data[i][9] == '310':
        positions.append((round(float(data[i][1]), 1), round(float(data[i][2]), 1)))
print(positions)