'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python code for simulations on car-like robot using T-lqr method.
'''
#!/usr/bin/env python
from __future__ import division
import h5py
from casadi import *
from T_LQR_car import T_LQR_car
import matplotlib.pyplot as plt
import numpy as np
import car_sims.params as params

#Initial state
X_0 = DM([0, 0, 0, 0]) # Initial state
x_g = DM([5.0, 5.0, 0, 0]) # goal state

#state dimension
n_x = params.n_x
#control imension
n_u = params.n_u
#horizon
horizon = params.horizon

control_upper_bound = DM([params.r_u[0], params.r_w[0]])
control_lower_bound = DM([params.r_u[1], params.r_w[1]])

#use the T_lqr class
t_lqr = T_LQR_car(n_x, n_u, horizon, X_0, x_g, control_upper_bound, control_lower_bound, params.dt)

# execute the algorithm
t_lqr.run_t_lqr()

t_lqr.plot_position(t_lqr.X_o)


'''
#save the trajectory
f = open('Tlqr_no_limit.txt','a')
for i in range(len(t_lqr.X_p)):
	f.write(str(t_lqr.X_o[i][0][0])+ '\t'+ str(t_lqr.X_o[i][1][0]) + '\t' + str(t_lqr.X_o[i][2][0]) + '\t' + str(t_lqr.X_o[i][3][0])+'\t'+ str(t_lqr.U_o[i][0][0])+'\t'+ str(t_lqr.U_o[i][1][0])+'\n')
f.close()
'''

#initialize the scaling factor for noise

epsilon = 0
epsilon_max = 0.1

#delta - increment in epsilon for sims
delta = .005

#no. of sims per epsilon
n_sims = 100

#creating trajectory variables to store the entire trajectory
X_t, U_t = t_lqr.create_traj_variables_DM()


while epsilon <= epsilon_max:

	cost_array = []

	for times in range(0, n_sims):
		
	
		for t in range(0, horizon):

			#apply the controller
			U_t[t] = t_lqr.U_o[t] + (0 if t==0 else 1) * mtimes(t_lqr.K_o[t-1], (X_t[t-1] - t_lqr.X_o[t-1]))

			if t==0:

				X_t[t] = t_lqr.car_like_dynamics_propagation_d_noisy(X_0, U_t[0], epsilon)

			else:

				X_t[t] = t_lqr.car_like_dynamics_propagation_d_noisy(X_t[t-1], U_t[t], epsilon)


		cost = t_lqr.calculate_total_cost(X_0, X_t, U_t, horizon)			

		cost_array.append(cost)

	with h5py.File('cost_data.hdf5','a') as f:

		dataset = f.create_dataset("{}".format(epsilon), data=cost_array)
			
	epsilon += delta



