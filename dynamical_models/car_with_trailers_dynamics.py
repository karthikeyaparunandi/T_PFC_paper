'''
	Dynamics of car-like model with  2 trailers attached with 2 controls and 6 state variables
	l_cr is the distance between the end of a trailer and the midpoint of its following trailer
	Karthikeya Parunandi
	Date: 2/20/19
'''
from __future__ import division

import numpy as np
from scipy.integrate import odeint
from casadi import *
import math


class car_w_trailers(object):

	def __init__(self, L, l_cr, dt):

		self.u_t = [0,0]
		self.l_cr = l_cr
		self.dt = dt
		self.L = L
	def car_w_trailers_dynamics_propagation(self, x, u, t):

		self.u_t = u

		x_next = odeint(self.dynamics, np.reshape(x, (6,)), np.linspace(t, t+self.dt, 2))
		x_next = np.array(x_next[1]).flatten()

		return 	x_next

	def car_w_trailers_dynamics_propagation_d(self, x, u):

		x_next = blockcat([[x[0] + self.dt*cos(x[2])*(u[0])],\
			 [x[1] + self.dt*sin(x[2])*(u[0])], \
			 	[x[2] + self.dt*(u[0])*tan(x[3])/self.L], \
			 		[x[3] + self.dt*(u[1])], \
			 			[x[4] + self.dt*(u[0])*sin(x[2] - x[4])/self.l_cr[0]], \
			 				[x[5] + (u[0])*self.dt*cos(x[2]-x[4])*sin(x[4] - x[5])/self.l_cr[1] ]])

		
		return 	x_next

	def car_w_trailers_dynamics_propagation_d_noisy(self, x, u, epsilon):
		# process noise added to dynamics
		w = epsilon*np.sqrt(self.dt)*np.random.normal(0.0, 1.0, 6)

		x_next = blockcat([[x[0] + self.dt*cos(x[2])*u[0]+w[0]],\
			 [x[1] + self.dt*sin(x[2])*u[0]+w[1]], \
			 	[x[2] + self.dt*(u[0])*tan(x[3])/self.L+w[2]], \
			 		[x[3] + self.dt*u[1]+w[3]], \
			 			[x[4] + self.dt*u[0]*sin(x[2] - x[4])/self.l_cr[0]+w[4]], \
			 				[x[5] + u[0]*self.dt*cos(x[2]-x[4])*sin(x[4] - x[5])/self.l_cr[1] +w[5]]])

		return 	x_next

	def dynamics(self, x, t):

		#g = blockcat([[cos(x[2]), 0], [sin(x[2]),0], [tan(x[3])/self.L, 0], [0, 1]])
		x_dot = mtimes(g, self.u_t)
		return np.reshape(x_dot,(6,))
		