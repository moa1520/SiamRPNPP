import numpy as np
import matplotlib.pyplot as plt

f = open('ground_truth/Non_video4_Final_result_iou.txt', 'r')

output = f.readline()
output = output.split(',')
output = output[:len(output)-1]

output = list(map(float, output))
output = np.asarray(output)
print("평균 :", output.mean())
f.close()

x_values = list(range(1, len(output) + 1))

plt.plot(x_values, output)
plt.grid()
plt.show()
