'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python code for simulations on car-like robot using T-PFC method.
'''
#!/usr/bin/env python
from __future__ import division
import h5py
from casadi import *
from DDP_car import DDP_car
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


#use DDP class
ddp = DDP_car(n_x, n_u, horizon, X_0, x_g, control_upper_bound, control_lower_bound, params.dt)


#perform ddp iterations
ddp.iterate_ddp()

ddp.plot_position()


#initialize the scaling factor for noise
epsilon = 0
epsilon_max = 0.1

#delta - increment in epsilon for sims
delta = .005

#no. of sims per epsilon
n_sims = 50

#creating trajectory variables to store the entire trajectory
X_t, U_t = ddp.create_traj_variables_DM()

while epsilon <= epsilon_max:

	cost_array = []

	for times in range(0, n_sims):
		
	
		for t in range(0, horizon):

			#apply the controller
			U_t[t] = ddp.U_p[t] + (0 if t==0 else 1) * mtimes(ddp.K[t-1], (X_t[t-1] - ddp.X_p[t-1]))

			if t==0:

				X_t[t] = ddp.car_like_dynamics_propagation_d_noisy(X_0, U_t[0], epsilon)

			else:

				X_t[t] = ddp.car_like_dynamics_propagation_d_noisy(X_t[t-1], U_t[t], epsilon)


		cost = ddp.calculate_total_cost(X_0, X_t, U_t, horizon)			

		cost_array.append(cost)

	with h5py.File('cost_data.hdf5','a') as f:

		dataset = f.create_dataset("{}".format(epsilon), data=cost_array)
			
	epsilon += delta
