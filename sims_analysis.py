from __future__ import division

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import brewer2mpl


t_pfc = 'car_sims/T_PFC/cost_data.hdf5' 
ddp = 'car_sims/DDP/cost_data.hdf5'
t_lqr = 'car_sims/T_LQR/cost_data.hdf5'
'''
# car with trailers

t_pfc = 'car_with_trailers_sims/T_PFC/cost_data.hdf5' 
ddp = 'car_with_trailers_sims/DDP/cost_data.hdf5'
t_lqr = 'car_with_trailers_sims/T_LQR/cost_data.hdf5'
'''
# T-PFC
cost_mean_tpfc = []
cost_std_tpfc = []
epsilons_t_pfc = []

f = h5py.File(t_pfc,'r+')
prune = u'055'

for key in f.keys():
	if key < u'0.13':
		data = f[key]
		if key == u'0':
			nominal = np.mean(data.value)
		#print(data.value, key)		
		if np.mean(data.value) < 1e+50:
			epsilons_t_pfc.append(key)
			cost_mean_tpfc.append(np.mean(data.value)/nominal)
			cost_std_tpfc.append(np.std(data.value)/nominal)
		
		if key == prune:
			for i in range(len(data.value)):
				if data[i] > 2e+60:
					data[i] = data[i-1]

f.close()


# DDP
cost_mean_ddp = []
cost_std_ddp =[]
epsilons_ddp = []

f = h5py.File(ddp,'r+')

for key in f.keys():
	if key < u'0.13':
	
		data = f[key]
		if key == u'0':
			nominal = np.mean(data.value)

		#print(data.value, key)
		if np.mean(data.value) < 1e+50:
			epsilons_ddp.append(key)
			cost_mean_ddp.append(np.mean(data.value)/nominal)
			cost_std_ddp.append(np.std(data.value)/nominal)
		
		if key == prune:
			
			for i in range(len(data.value)):
				if data[i] > 1e+6:
					data[i] = data[i]         
		

f.close()




# T-LQR
cost_mean_tlqr = []
cost_std_tlqr = []
epsilons_t_lqr = []

f = h5py.File(t_lqr,'r+')

for key in f.keys():
	if key < u'0.13':
		
		data = f[key]
		if key == u'0':
			nominal = np.mean(data.value)

		if np.mean(data.value) < 1e+50:
			epsilons_t_lqr.append(key)
			cost_mean_tlqr.append(np.mean(data.value)/nominal)
			cost_std_tlqr.append(np.std(data.value)/nominal)
		
		if key == prune:
			for i in range(len(data.value)):
				if data[i] > 1e+60:
					print("hi")
					data[i] = data[i]



f.close()



# MPC
err, eps, t, J_vec, J_ideal_vec, J_std, J_ideal_std, n_samples = np.genfromtxt('car_sims/MPC/data_mpc.txt', delimiter="\t", usecols=(0,1,2,3,4,5,6,7)).T
print(J_vec[0])
# ---------------------------------------------------
# PLOTTING RESULTS


# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors
#mpl.pyplot.figure(dpi=600)

params = {
   'axes.labelsize': 10,
   'font.size': 8,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4, 3], # instead of 4.5, 4.5
   'font.weight': 'bold',
   }
   
mpl.rcParams.update(params)
# T-PFC
plt.plot(epsilons_t_pfc, cost_mean_tpfc, linewidth=2,color=colors[0] )
plt.fill_between(epsilons_t_pfc, np.array(cost_mean_tpfc)+np.array(cost_std_tpfc), np.array(cost_mean_tpfc)-np.array(cost_std_tpfc), alpha=0.35, linewidth=0, color=colors[0])
plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

# DDP
plt.plot(epsilons_ddp, cost_mean_ddp, linewidth=2, color=colors[1])
plt.fill_between(epsilons_ddp, np.array(cost_mean_ddp)+np.array(cost_std_ddp), np.array(cost_mean_ddp)-np.array(cost_std_ddp), alpha=0.25, linewidth=0, color=colors[1])
plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

# T-LQR
plt.plot(epsilons_t_lqr, cost_mean_tlqr, linewidth=2, color=colors[2])
plt.fill_between(epsilons_t_lqr, np.array(cost_mean_tlqr)+np.array(cost_std_tlqr), np.array(cost_mean_tlqr)-np.array(cost_std_tlqr), alpha=0.25, linewidth=0, color=colors[2])
plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

#MPC
print(epsilons_ddp)
plt.plot(epsilons_ddp, J_vec/J_vec[0], linewidth=2, color=colors[3])
plt.fill_between(epsilons_ddp, (J_vec + J_std)/J_vec[0], (J_vec - J_std)/J_vec[0], alpha=0.3, linewidth=0, color=colors[3])
plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
plt.grid(axis='x', color="0.9", linestyle='-', linewidth=1)

print(epsilons_t_pfc, eps.tolist())
plt.legend(['T-PFC', 'ILQG', 'T-LQR', 'MPC'])


plt.xlabel('$\epsilon$')
plt.ylabel('Cost incurred (nominal scaled to 1)')

locs, labs = plt.xticks()

plt.xticks( np.arange(0, 70, step=4))
#
#plt.legend("Cost mean of DDP","Cost mean of T-PFC")

plt.show()



