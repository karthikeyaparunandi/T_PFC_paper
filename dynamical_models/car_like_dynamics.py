'''
	Dynamics of car-like model with 2 controls and 4 state variables
	Karthikeya Parunandi
	Date: 2/11/19
'''
from __future__ import division

import numpy as np
from scipy.integrate import odeint
from casadi import *
import math


class car_like(object):

	def __init__(self, L, dt):

		self.u_t = [0,0]
		self.L = L
		self.dt = dt

	def car_like_dynamics_propagation(self, x, u, t):

		self.u_t = u

		x_next = odeint(self.dynamics, np.reshape(x, (4,)), np.linspace(t, t+self.dt, 2))
		x_next = np.array(x_next[1]).flatten()

		return 	x_next

	def car_like_dynamics_propagation_d(self, x, u):

		x_next = blockcat([[x[0]+ self.dt*cos(x[2])*u[0]], [x[1] + self.dt*sin(x[2])*u[0]], [x[2] + ((u[0]*tan(x[3])*self.dt)/self.L)], [x[3] + self.dt*u[1]]])

		return 	x_next

	def car_like_dynamics_propagation_d_noisy(self, x, u, epsilon):
		# process noise added to dynamics
		w = epsilon*np.random.normal(0.0, 1.0, 4)

		x_next = blockcat([[x[0]+ self.dt*cos(x[2])*u[0] + np.sqrt(self.dt)*w[0]], [x[1] + self.dt*sin(x[2])*u[0]+ np.sqrt(self.dt)*w[1]], [x[2] + ((u[0]*tan(x[3])*self.dt)/self.L)+ np.sqrt(self.dt)*w[2]], \
				[x[3] + self.dt*u[1]+ np.sqrt(self.dt)*w[3]]])

		return 	x_next

	def dynamics(self, x, t):

		g = blockcat([[cos(x[2]), 0], [sin(x[2]),0], [tan(x[3])/self.L, 0], [0, 1]])
		x_dot = mtimes(g, self.u_t)
		return np.reshape(x_dot,(4,))
		