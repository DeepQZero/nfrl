import numpy as np

data = np.load('week1.npy')
print(data[0:5])

vals = []
for row in data:
    if row[1] != 'x':
        vals.append(float(row[2]))

print(min(vals))
print(max(vals))