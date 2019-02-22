'''
 Written by Karthikeya Parunandi
 Date - 2/11/19

 Brief description : This is an implementation of Differential Dynamic Programming (control-limited) using a class.
 The paper based on which implementation is done is "https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf"
'''


# Optimization library - though no optimization involved, this is used to import DM, jacobian and hessian calculation functionalities from casadi
from __future__ import division

from casadi import *
# Numerics
import numpy as np
import math
# Parameters
import car_with_trailers_sims.params as params
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
from algorithm_classes.DDP import DDP 
from dynamical_models.car_with_trailers_dynamics import car_w_trailers

class DDP_car_w_trailers(DDP, car_w_trailers):

	def __init__(self, n_x, n_u, horizon, initial_state, final_state, control_ub, control_lb, dt):

		self.X_0 = initial_state
		self.X_g = final_state
		self.dt = dt
		self.L = params.L

		DDP.__init__(self, n_x, n_u, horizon, initial_state, final_state, control_ub, control_lb)
		car_w_trailers.__init__(self, params.L,params.l_cr, dt)

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


	def plot_position(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# obstacle patches
		# obstacle patches
		patch_1 = PolygonPatch(params.obstacle_1, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		#patch_2 = PolygonPatch(params.obstacle_2, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		#patch_3 = PolygonPatch(params.obstacle_3, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
	
		ax.add_patch(patch_1)

		x = [self.X_p[i][0] for i in range(len(self.X_p))]
		y = [self.X_p[i][1] for i in range(len(self.X_p))]

		plt.plot(x, y, '-')
		plt.show()

	def create_traj_variables_DM(self):

		X_t = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		U_t = horzsplit(DM(self.n_u, self.N), self.Sl_traj)

		return X_t, U_t




