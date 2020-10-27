import numpy as np
import matplotlib.pyplot as plt

f = open('ground_truth/120result_iou.txt', 'r')
f2 = open('ground_truth/120result_pre.txt', 'r')

output = f.readline()
output = output.split(',')
output = output[:len(output)-1]

output = list(map(float, output))
output = np.asarray(output)
print("평균 :", output.mean())
f.close()

pre = f2.readline()
pre = pre.split(',')
pre = pre[:len(pre)-1]
pre = list(map(float, pre))
pre = np.asarray(pre)
print("평균 :", pre.mean())
f2.close()

x_values = list(range(1, len(output) + 1))
fig, ax1 = plt.subplots()
line1 = ax1.plot(x_values, output, color='C0',
                 label='IoU')

ax2 = ax1.twinx()
line2 = ax2.plot(x_values, pre, label='Distance', color='C2')

lines = line1 + line2
labels = [i.get_label() for i in lines]

ax1.set_xlabel('Frame')
ax1.set_ylabel('Intersection over Union')
ax2.set_ylabel('Distance')
# plt.xlabel('Frame')
ax1.legend(lines, labels, loc='center right')
plt.show()
