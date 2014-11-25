import scipy.io as sio

mat = sio.loadmat("costsmooth.mat")
mat_cost= mat["cost"]
max_size = mat_cost.size
count_epoch = 100
window = 100.
grad = []

while count_epoch < max_size :
    grad.append(mat_cost[count_epoch] - mat_cost[count_epoch - window])
    count_epoch += window

sio.savemat('gradient', {'grad':grad})
