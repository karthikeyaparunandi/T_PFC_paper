'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python class for T-LQR method on car-like robot.
'''
#!/usr/bin/env python
from __future__ import division

from casadi import *
import matplotlib.pyplot as plt
import math 
import time
import numpy as np
import car_with_trailers_sims.params as params
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
#from rrt_planning import plan
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
from algorithm_classes.T_LQR import T_LQR
from dynamical_models.car_with_trailers_dynamics import car_w_trailers


class T_LQR_car_w_trailers(T_LQR, car_w_trailers):

	def __init__(self, n_x, n_u, horizon, initial_state, final_state, control_upper_bound, control_lower_bound, dt):

		self.X_0 = initial_state
		self.X_g = final_state
		self.dt = dt
		self.L = params.L

		self.W_x_LQR = params.W_x_LQR
		self.W_x_LQR_f = params.W_x_LQR_f
		self.W_u_LQR = params.W_u_LQR
		
		T_LQR.__init__(self, n_x, n_u, horizon, initial_state, final_state, control_upper_bound, control_lower_bound)
		car_w_trailers.__init__(self, params.L, params.l_cr, dt)

	def dynamics_propagation(self, state, control, time):

		state_next = self.car_w_trailers_dynamics_propagation(state, control, time)

		return state_next

	def dynamics_propagation_d(self, state, control):

		state_next = self.car_w_trailers_dynamics_propagation_d(state, control)

		return state_next

	def cost(self, x, u):

		# Defining objective function
			
		Cost_x = mtimes(mtimes((x - self.X_g).T, params.W_x), (x - self.X_g))
		for i in range(0, params.n_obstacles):
			Cost_x +=  blockcat([[exp(-params.pho[i]*mtimes(mtimes((x - params.x_obstacle[i]).T, params.Q_obs[i]), x - params.x_obstacle[i]) + params.pho[i]) + \
					exp(-params.pho[i]*mtimes(mtimes((x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]) , (params.l_cr[1]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]).T, params.Q_obs[i]), (x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]), (params.l_cr[0]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]))+ params.pho[i]) + \
					exp(-params.pho[i]*mtimes(mtimes((x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]) + (params.l_cr[1]+self.L/2)*cos(x[5]) , (params.l_cr[0]+self.L/2)*sin(x[4])+ (params.l_cr[1]+self.L/2)*sin(x[5]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]).T, params.Q_obs[i]), (x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[3]) + (params.l_cr[1]+self.L/2)*cos(x[4]), (params.l_cr[0]+self.L/2)*sin(x[3])+ (params.l_cr[1]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i])) + params.pho[i])]]) 
		
		cost = mtimes(mtimes(u.T, params.W_u), u) + Cost_x 


		return cost

	def cost_final(self, state):

		x = state

		cost_x = mtimes(mtimes((x - self.X_g).T, params.W_x_f), (x - self.X_g))
		for i in range(0, params.n_obstacles):
			cost_x +=  blockcat([[exp(-params.pho[i]*mtimes(mtimes((x - params.x_obstacle[i]).T, params.Q_obs[i]), x - params.x_obstacle[i]) + params.pho[i]) + \
					exp(-params.pho[i]*mtimes(mtimes((x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]) , (params.l_cr[1]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]).T, params.Q_obs[i]), (x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]), (params.l_cr[0]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]))+ params.pho[i]) + \
					exp(-params.pho[i]*mtimes(mtimes((x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[4]) + (params.l_cr[1]+self.L/2)*cos(x[5]) , (params.l_cr[0]+self.L/2)*sin(x[4])+ (params.l_cr[1]+self.L/2)*sin(x[5]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i]).T, params.Q_obs[i]), (x -  blockcat([[(params.l_cr[0]+self.L/2)*cos(x[3]) + (params.l_cr[1]+self.L/2)*cos(x[4]), (params.l_cr[0]+self.L/2)*sin(x[3])+ (params.l_cr[1]+self.L/2)*sin(x[4]), x[2], x[3], x[4], x[5]]]).T - params.x_obstacle[i])) + params.pho[i])]]) 
		
		return cost_x

	def initialize_traj(self):

		array = np.loadtxt('../rrt_path.txt')
		for t in range(0, self.N):
			self.X_p[t] = DM(array[t, 0:self.n_x])
			self.U_p[t] = DM(array[t, self.n_x:self.n_x+self.n_u])


	def plot_position(self, state_traj):

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# obstacle patches
		patch_1 = PolygonPatch(params.obstacle_1, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		#patch_2 = PolygonPatch(params.obstacle_2, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		#patch_3 = PolygonPatch(params.obstacle_3, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
	
		ax.add_patch(patch_1)
		#ax.add_patch(patch_2)
		#ax.add_patch(patch_3)
	
		x = [state_traj[i][0] for i in range(0, self.N)]#[self.X_o[i][0] for i in range(0, self.N)]
		y = [state_traj[i][1] for i in range(0, self.N)]#[self.X_o[i][1] for i in range(0, self.N)]

		plt.plot(x, y, '-')
		plt.show()

	def create_traj_variables_DM(self):

		X_t = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		U_t = horzsplit(DM(self.n_u, self.N), self.Sl_traj)

		return X_t, U_t
