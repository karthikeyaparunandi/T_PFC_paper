'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python class for T-PFC method.
'''
#!/usr/bin/env python
from __future__ import division
from casadi import *
import math 
import time
import numpy as np
#from rrt_planning import plan


class T_PFC(object):

	def __init__(self, n_x, n_u, horizon, initial_state, final_state, control_upper_bound, control_lower_bound):

		self.n_x = n_x
		self.n_u = n_u
		self.N = horizon

		#initialize optistack
		self.opti = Opti()	

		# trajectory slicing
		self.Sl_traj = self.slice_traj(1, self.N)
		# sensitivity matrix slicing
		self.Sl_sens = self.slice_traj(self.n_x, self.N)

		#  decision variables ----------------
		# state trajectory 
		self.X = horzsplit(self.opti.variable(self.n_x, self.N), self.Sl_traj)
		self.U = horzsplit(self.opti.variable(self.n_u, self.N), self.Sl_traj)

		# Define optimal nominal trajectory 
		self.X_o = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		self.U_o = horzsplit(DM(self.n_u, self.N), self.Sl_traj)
		self.K_o = horzsplit(DM(self.n_u, self.n_x*self.N), self.Sl_sens)

		# create variables to store initial guesses for optimization
		self.X_p = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		self.U_p = horzsplit(DM(self.n_u, self.N), self.Sl_traj)


		# define generalized variables to put in a function
		self.x= MX.sym('x', self.n_x, 1)
		self.u= MX.sym('u', self.n_u, 1)

		#control limits
		self.u_lb = control_lower_bound
		self.u_ub = control_upper_bound

		# initialize the execution time of t_pfc algorithm 
		self.t_pfc_execution_time  = 0


	def run_t_pfc(self):

		start_time = time.time()
		self.define_partial_func()
		self.initialize_traj()
		self.solve_OCP()
		self.backward_pass()
		self.t_pfc_execution_time = time.time() - start_time
		#print(self.X_0, self.X_o[0][0][0],self.cost(self.X_0, self.X_o[0][0][0]))

	def solve_OCP(self):

		#Define objective function
		Obj = self.calculate_total_cost(self.X_0, self.X, self.U, self.N)

		# Minimize the objective function
		self.opti.minimize(Obj)

		# specify the constraints
		for t in range(0, self.N):

			# dynamics propagation
			if t == 0:
				self.opti.subject_to(self.X[0] == self.dynamics_propagation_d(self.X_0, self.U[0]))
			else:
				self.opti.subject_to(self.X[t] == self.dynamics_propagation_d(self.X[t-1], self.U[t]))

			# Bounds of U[t]
			for i in range(0, self.n_u):
				self.opti.subject_to(self.opti.bounded(self.u_lb[i] ,self.U[t][i], self.u_ub[i]))
			
		# Initializing U[t] and X[t] with the initial guesses
			replan_t = 0
			if replan_t == 0:
			
				for j in range(0, self.n_u):
				
					self.opti.set_initial(self.U[t][j], self.U_p[t][j])
				
				for j in range(0, self.n_x):

					self.opti.set_initial(self.X[t][j], self.X_p[t][j])
					
			else:

				for j in range(0, self.n_u):
				
					self.opti.set_initial(self.U[t][j], U_o[t+replan_t][j])
				
				for j in range(0, self.n_x):

					self.opti.set_initial(self.X[t][j], X_o[t+replan_t][j])
			

		opts = {}
		opts['ipopt.print_level'] = 0
		self.opti.solver("ipopt", opts) # set numerical backend
		sol = self.opti.solve()   # actual solve		# ---- solve NLP              ------
		print("cost incurred:", sol.value(Obj))
		self.X_o = [np.reshape(sol.value(self.X[l]), (self.n_x, 1)) for l in range(0, self.N)]
		self.U_o = [np.reshape(sol.value(self.U[l]), (self.n_u, 1)) for l in range(0, self.N)]


	def backward_pass(self,):

		R =  self.l_uu(self.X_o[self.N-2], self.U_o[self.N-1])

		G = self.l_x_f(self.X_o[self.N-1]) 
		P = 0.5*self.l_xx_f(self.X_o[self.N-1]) 


		for t in range(self.N-1, 0, -1):

			A_o = DM(self.f_x(self.X_o[t-1], self.U_o[t])) 
			B_o = DM(self.f_u(self.X_o[t-1], self.U_o[t]))

			S = 0.5*R + mtimes( mtimes(B_o.T, P), B_o)

			# T-PFC gain 
			self.K_o[t-1] = -mtimes(inv(S), mtimes( mtimes(B_o.T, P), A_o))	
			
			# first and second order equations
			G = self.l_x(self.X_o[t-1], self.U_o[t]) + mtimes(G, A_o)

			P = 0.5*DM(self.l_xx(self.X_o[t-1], self.U_o[t]))  +  mtimes(mtimes(A_o.T, P), A_o) - mtimes(mtimes(self.K_o[t-1].T, S), self.K_o[t-1]) +\
						 np.tensordot(G, np.reshape(self.f_xx(self.X_o[t-1], self.U_o[t]), (4,4,4)), axes=(1,1))[0] 



	def define_partial_func(self):

		self.f_x = Function('f_x' , [self.x, self.u] , [jacobian(self.dynamics_propagation_d(self.x, self.u) , self.x)])
		self.f_u = Function( 'f_u' , [self.x, self.u] , [jacobian(self.dynamics_propagation_d(self.x, self.u), self.u)])
		self.f_xx = Function( 'f_xx' , [self.x, self.u] , [jacobian(jacobian(self.dynamics_propagation_d(self.x, self.u) , self.x), self.x)])


		self.l_x = Function('l_x', [self.x, self.u], [jacobian(self.cost(self.x, self.u) , self.x)])
		self.l_x_f = Function('l_x_f', [self.x], [jacobian(self.cost_final(self.x) , self.x)])

		self.l_xx = Function('l_xx', [self.x, self.u], [jacobian(jacobian(self.cost(self.x, self.u) , self.x), self.x)])
		self.l_xx_f = Function('l_xx_f', [self.x], [jacobian(jacobian(self.cost_final(self.x) , self.x), self.x)])

		#self.l_u = Function('l_u', [self.x, self.u], [jacobian(self.cost(self.x, self.u), self.u)])
		self.l_uu = Function('l_uu', [self.x, self.u], [jacobian(jacobian(self.cost(self.x, self.u), self.u), self.u)])



	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		#initialize cost_total with initial cost
		cost_total = self.cost(initial_state, control_traj[0])

		for t in range(0, horizon-1):

			cost_total += self.cost(state_traj[t], control_traj[t+1])

		if horizon == self.N:
			cost_total += self.cost_final(state_traj[horizon-1])
		else:
			cost_total += self.cost(state_traj[horizon-1], control_traj[horizon])

		return cost_total

	def slice_traj(self, n_columns, horizon):

		'''
		horizon - specify the horizon of the trajectory in terms of time-steps
		n_columns - no. of columns in the given variable at a time-step. for ex. - state and control have 1 column each
		'''
		Sl = []
		for l in range(0, horizon + 1):
			Sl.append(n_columns*l)

		return Sl

	def cost(self, state, control):

		raise NotImplementedError()

	def dynamics_propagation(self, state, control, time):

		# propagate dynamics forward
		raise NotImplementedError()

	def dynamics_propagation_d(self, state, control):

		# propagate dynamics forward
		raise NotImplementedError()

	def initialize_traj(self):
		# initial guess for the trajectory
		pass


