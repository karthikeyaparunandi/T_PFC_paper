'''
	Dynamics of car-like model with 2 controls i.e, (force and torque) and 12 state variables
	Look at constrained DDP paper for reference - http://motion.pratt.duke.edu/papers/ICRA2017-Xie-ConstrainedDDP.pdf
	Karthikeya Parunandi
	Date: 2/22/19
'''
from __future__ import division

import numpy as np
from scipy.integrate import odeint
from casadi import *
import math


class quadrotor(object):

	def __init__(self, L, dt, I_c, g, k_d, m):

		self.u_t = [0,0]
		self.L = L
		self.dt = dt
		self.I_c = I_c 
		self.g = g #scalar - accln due to gravity 
		self.k_d = k_d
		self.m = m # mass

	def quadrotor_dynamics_propagation(self, x, u, t):

		self.u_t = u

		x_next = odeint(self.dynamics, np.reshape(x, (4,)), np.linspace(t, t+self.dt, 2))
		x_next = np.array(x_next[1]).flatten()

		return 	x_next

	def quadrotor_dynamics_propagation_d(self, x, u):
		'''
		u - controls, with u[0] being force and u[1] being torque in the body framer

		'''
		R_theta = self.Rotation_matrix(x[3:6])
		x_next = x + self.dt* [x[6:9], (self.g + (mtimes(R_theta, u[0]) - self.k_d*x[6:9]))/self.m, mtimes(inv(self.J_w), x[9:12]), mtimes(inv(self.I_c), u[1])] 

		return 	x_next

	def quadrotor_dynamics_propagation_d_noisy(self, x, u, epsilon):

		# process noise added to dynamics
		w = epsilon*np.random.normal(0.0, 1.0, 4)

		x_next = blockcat([[x[0]+ self.dt*cos(x[2])*u[0] + np.sqrt(self.dt)*w[0]], [x[1] + self.dt*sin(x[2])*u[0]+ np.sqrt(self.dt)*w[1]], [x[2] + ((u[0]*tan(x[3])*self.dt)/self.L)+ np.sqrt(self.dt)*w[2]], \
				[x[3] + self.dt*u[1]+ np.sqrt(self.dt)*w[3]]])

		return 	x_next

	def dynamics(self, x, t):

		g = blockcat([[cos(x[2]), 0], [sin(x[2]),0], [tan(x[3])/self.L, 0], [0, 1]])
		x_dot = mtimes(g, self.u_t)
		return np.reshape(x_dot,(4,))
	
	def Rotation_matrix(self, angles_vec):

		# Rotation fromthe body frame to inertial frame
		phi = angles_vec[0] # roll
		theta = angles_vec[1] # pitch
		psi = angles_vec[2]	#yaw
		
		R_theta = DM([[cos(psi)*cos(theta), cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi), cos(psi)*sin(theta)*cos(phi) + sin(phi)*sin(psi) ], []\
					 [sin(psi)*cos(theta), sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi), sin(psi)*sin(theta)*cos(phi) - sin(phi)*cos(psi) ],\
					 [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])

		return R_theta

	def euler_rate_to_w(self, angles_dot_vec):

		phi_d = angles_dot_vec[0] # roll
		theta_d = angles_dot_vec[1] # pitch
		psi_d = angles_dot_vec[2]	#yaw
		W = DM([phi_d - sin(theta)*psi_d, cos(phi)*theta_d + cos(theta)*sin(phi)*psi_d, -sin(phi)*theta_d + cos(theta)*cos(phi)*phi_d])

		return W