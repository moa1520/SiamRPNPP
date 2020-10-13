import numpy as np

f = open('ground_truth/result_cls.txt', 'r')

while True:
    line = f.readline()
    if not line:
        break
    output = line.split(',')

output = list(map(float, output[:len(output)-1]))
output = np.asarray(output)
print(output.mean())
f.close()
