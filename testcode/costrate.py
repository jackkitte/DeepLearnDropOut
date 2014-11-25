import scipy.io as sio
import numpy as np

mat = sio.loadmat("cost.mat")
mat_cost = mat["cost"]
max_size = mat_cost.size
count_epoch = 1000.
count_size = 500.
costrate_size = (max_size / count_size) - 2
costrate_mat = np.zeros(costrate_size)
costrate_epoch = 0

while count_epoch < max_size :
    costrate = np.average(mat_cost[(count_epoch - count_size):count_epoch]) / np.average(mat_cost[(count_epoch - (2 * count_size)):(count_epoch - count_size)])
    costrate_mat[costrate_epoch] = costrate
    count_epoch += count_size
    costrate_epoch += 1

sio.savemat('costrate', {'costrate':costrate_mat})
