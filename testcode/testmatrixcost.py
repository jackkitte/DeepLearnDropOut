import scipy.io as sio

mat = sio.loadmat("cost.mat")
mat_cost= mat["cost"]
max_size = mat_cost.size
count_epoch = 0
window = 51
mat_cost_re = []

while count_epoch < max_size:
    if count_epoch < (window / 2.):
        mat_cost_re.append(mat_cost[count_epoch])
        count_epoch += 1
    elif (max_size - window / 2.) < count_epoch:
        mat_cost_re.append(mat_cost[count_epoch])
        count_epoch += 1
    else:
        mat_cost_re.append(sum(mat_cost[count_epoch - (window / 2):count_epoch + (window / 2) + 1]) / window)
        count_epoch += 1

sio.savemat('costsmooth', {'cost':mat_cost_re})
