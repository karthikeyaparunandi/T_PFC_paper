'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python code for simulations on car-like robot using T-PFC method.
'''
#!/usr/bin/env python
from __future__ import division
import h5py
from casadi import *
from T_PFC_car_w_trailers import T_PFC_car_w_trailers
import matplotlib.pyplot as plt
import numpy as np
import car_with_trailers_sims.params as params

#Initial position
X_0 = DM([0, 0, 0, 0, 0, 0]) # Initial state
x_g = DM([5.0, 6.0, 0, 0, 0, 0]) # goal state

#state dimension
n_x = params.n_x
#control imension
n_u = params.n_u
#horizon
horizon = params.horizon

control_upper_bound = DM([params.r_u[0], params.r_w[0]])
control_lower_bound = DM([params.r_u[1], params.r_w[1]])

#use the T_PFC class
t_pfc = T_PFC_car_w_trailers(n_x, n_u, horizon, X_0, x_g, control_upper_bound, control_lower_bound, params.dt)

# execute the algorithm
t_pfc.run_t_pfc()

t_pfc.plot_position(t_pfc.X_o)

print(t_pfc.K_o)
'''
#save the trajectory
f = open('TPFC_no_limit.txt','a')
for i in range(len(t_pfc.X_p)):
	f.write(str(t_pfc.X_o[i][0][0])+ '\t'+ str(t_pfc.X_o[i][1][0]) + '\t' + str(t_pfc.X_o[i][2][0]) + '\t' + str(t_pfc.X_o[i][3][0])+'\t'+ str(t_pfc.U_o[i][0][0])+'\t'+ str(t_pfc.U_o[i][1][0])+'\n')
f.close()

'''
'''
#initialize the scaling factor for noise
epsilon = 0
epsilon_max = 0.1

#delta - increment in epsilon for sims
delta = .005

#no. of sims per epsilon
n_sims = 100

#creating trajectory variables to store the entire trajectory
X_t, U_t = t_pfc.create_traj_variables_DM()


while epsilon <= epsilon_max:

	cost_array = []

	for times in range(0, n_sims):
		
	
		for t in range(0, horizon):

			#apply the controller
			U_t[t] = t_pfc.U_o[t] + (0 if t==0 else 1) * mtimes(t_pfc.K_o[t-1], (X_t[t-1] - t_pfc.X_o[t-1]))

			if t==0:

				X_t[t] = t_pfc.car_w_trailers_dynamics_propagation_d_noisy(X_0, U_t[0], epsilon)

			else:

				X_t[t] = t_pfc.car_w_trailers_dynamics_propagation_d_noisy(X_t[t-1], U_t[t], epsilon)


		cost = t_pfc.calculate_total_cost(X_0, X_t, U_t, horizon)			

		cost_array.append(cost)

	with h5py.File('cost_data.hdf5','a') as f:

		dataset = f.create_dataset("{}".format(epsilon), data=cost_array)
			
	epsilon += delta
'''


