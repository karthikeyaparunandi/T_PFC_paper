from __future__ import division

import numpy as np
import h5py
import matplotlib.pyplot as plt

t_pfc = '/car_sims/T_PFC/cost_data.hdf5' 
ddp = '/car_sims/DDP/cost_data.hdf5'

cost_mean_tpfc = []
with h5py.File(t_pfc,'r') as f:
	for key in f.keys():
		data = f[key]
		#print(data.value, key)
		if np.mean(data.value) < 1e+6:
			cost_mean_tpfc.append(np.mean(data.value))

cost_mean_ddp = []

with h5py.File(ddp,'r') as f:
	for key in f.keys():
		data = f[key]
		#print(data.value, key)
		if np.mean(data.value) < 1e+6:
			cost_mean_ddp.append(np.mean(data.value))


#plt.plot(cost_mean_ddp)
plt.plot(cost_mean_tpfc)

plt.legend("Cost mean of DDP","Cost mean of T-PFC")

plt.show()
