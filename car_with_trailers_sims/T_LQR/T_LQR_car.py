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
import car_sims.params as params
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
#from rrt_planning import plan
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
from algorithm_classes.T_LQR import T_LQR
from dynamical_models.car_like_dynamics import car_like


class T_LQR_car(T_LQR, car_like):

	def __init__(self, n_x, n_u, horizon, initial_state, final_state, control_upper_bound, control_lower_bound, dt):

		self.X_0 = initial_state
		self.X_g = final_state
		self.dt = dt

		self.W_x_LQR = params.W_x_LQR
		self.W_x_LQR_f = params.W_x_LQR_f
		self.W_u_LQR = params.W_u_LQR

		T_LQR.__init__(self, n_x, n_u, horizon, initial_state, final_state, control_upper_bound, control_lower_bound)
		car_like.__init__(self, params.L, dt)

	def dynamics_propagation(self, state, control, time):

		state_next = self.car_like_dynamics_propagation(state, control, time)

		return state_next

	def dynamics_propagation_d(self, state, control):

		state_next = self.car_like_dynamics_propagation_d(state, control)

		return state_next

	def cost(self, x, u):

		cost = mtimes(mtimes((x - self.X_g).T, params.W_x), (x - self.X_g)) + \
				mtimes(mtimes(u.T, params.W_u), u) + \
				exp(-params.pho_1*mtimes(mtimes((x - params.x_obstacle_1).T, params.Q_1), x - params.x_obstacle_1) + params.pho_1) + \
				exp(-params.pho_2*mtimes(mtimes((x - params.x_obstacle_2).T, params.Q_2), x - params.x_obstacle_2) + params.pho_2) + \
				exp(-params.pho_3*mtimes(mtimes((x - params.x_obstacle_3).T, params.Q_3), x - params.x_obstacle_3) + params.pho_3) + \
				exp(-params.pho_4*mtimes(mtimes((x - params.x_obstacle_4).T, params.Q_4), x - params.x_obstacle_4) + params.pho_4) + \
				exp(-params.pho_5*mtimes(mtimes((x - params.x_obstacle_5).T, params.Q_5), x - params.x_obstacle_5) + params.pho_5)
	
		return cost

	def cost_final(self, state):

		x = state

		cost = mtimes(mtimes((x - self.X_g).T, params.W_x_f), (x - self.X_g)) + \
				exp(-params.pho_1*mtimes(mtimes((x - params.x_obstacle_1).T, params.Q_1), x - params.x_obstacle_1) + params.pho_1) + \
				exp(-params.pho_2*mtimes(mtimes((x - params.x_obstacle_2).T, params.Q_2), x - params.x_obstacle_2) + params.pho_2) + \
				exp(-params.pho_3*mtimes(mtimes((x - params.x_obstacle_3).T, params.Q_3), x - params.x_obstacle_3) + params.pho_3) + \
				exp(-params.pho_4*mtimes(mtimes((x - params.x_obstacle_4).T, params.Q_4), x - params.x_obstacle_4) + params.pho_4) + \
				exp(-params.pho_5*mtimes(mtimes((x - params.x_obstacle_5).T, params.Q_5), x - params.x_obstacle_5) + params.pho_5)

		return cost

	def initialize_traj(self):

		array = np.loadtxt('../rrt_path.txt')
		for t in range(0, self.N):
			self.X_p[t] = DM(array[t, 0:4])
			self.U_p[t] = DM(array[t, 4:6])


	def plot_position(self, state_traj):

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# obstacle patches
		patch_1 = PolygonPatch(params.obstacle_1, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		patch_2 = PolygonPatch(params.obstacle_2, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		patch_3 = PolygonPatch(params.obstacle_3, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		patch_4 = PolygonPatch(params.obstacle_4, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		patch_5 = PolygonPatch(params.obstacle_5, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
		patch_6 = PolygonPatch(params.obstacle_6, facecolor=[0,0.1,0.5], edgecolor=[0,0,0], alpha=0.7, zorder=2)
	
		ax.add_patch(patch_1)
		ax.add_patch(patch_2)
		ax.add_patch(patch_3)
		ax.add_patch(patch_4)
		ax.add_patch(patch_5)
		ax.add_patch(patch_6)

		x = [state_traj[i][0] for i in range(0, self.N)]#[self.X_o[i][0] for i in range(0, self.N)]
		y = [state_traj[i][1] for i in range(0, self.N)]#[self.X_o[i][1] for i in range(0, self.N)]

		plt.plot(x, y, '-')
		plt.show()

	def create_traj_variables_DM(self):

		X_t = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		U_t = horzsplit(DM(self.n_u, self.N), self.Sl_traj)
 
		return X_t, U_t
