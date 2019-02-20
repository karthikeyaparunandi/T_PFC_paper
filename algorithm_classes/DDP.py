'''
 Written by Karthikeya Parunandi
 Date - 2/10/19

 Brief description : This is an implementation of Differential Dynamic Programming (control-limited) using a class.
 The paper based on which implementation is done is "https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf"
'''


# Optimization library - though no optimization involved, this is used to import DM, jacobian and hessian calculation functionalities from casadi
from __future__ import division
from casadi import *
# Numerics
import numpy as np
import math
import copy
import time
# Parameters
#import params



class DDP(object):

	def __init__(self, n_x, n_u, horizon, initial_state, final_state, control_ub, control_lb):

		self.X_p_0 = initial_state
		self.X_g = final_state

		self.n_x = n_x
		self.n_u = n_u
		self.N = horizon


		self.alpha = .16

		self.Sl_traj = self.slice_traj(1, self.N)
		self.Sl_sens = self.slice_traj(self.n_x, self.N)

		# Define nominal state trajectory
		self.X_p = horzsplit(DM(self.n_x, self.N), self.Sl_traj)
		self.X_p_temp = horzsplit(DM(self.n_x, self.N), self.Sl_traj)

		# Define nominal control trajectory
		self.U_p = horzsplit(DM(self.n_u, self.N), self.Sl_traj)
		self.U_p_temp = horzsplit(DM(self.n_u, self.N), self.Sl_traj)

		# Define sensitivity matrices
		self.K = horzsplit(DM(self.n_u, self.n_x*self.N), self.Sl_sens)
		self.k = horzsplit(DM(self.n_u, self.N), self.Sl_traj)
		self.k_copy= copy.deepcopy(self.k)
		self.V_xx = horzsplit(DM(self.n_x, self.n_x*self.N), self.Sl_sens)
		self.V_x = horzsplit(DM(self.n_x, self.N), self.Sl_traj)

		# define generalized variables to put in a function
		self.x= MX.sym('x', self.n_x, 1)
		self.u= MX.sym('u', self.n_u, 1)

		# regularization parameter
		self.mu_min = 2
		self.mu = 1000	#10**(-6)
		self.mu_max = 10**(8)
		self.delta_0 = 2
		self.delta = 1	#self.delta_0
		self.c_1 = -1e-1
		self.u_ub = control_ub
		self.u_lb = control_lb
		self.count = 0

	def iterate_ddp(self):

		self.initialize_traj()
		
		#define partial functions
		self.define_partial_func()

		for j in range(150):


			b_pass_success_flag, del_J_alpha = self.backward_pass()


			if b_pass_success_flag == 1:

				self.regularization_dec_mu()
				f_pass_success_flag = self.forward_pass(del_J_alpha)
				#print(f_pass_success_flag, j,"th - iteration successful")
				if not f_pass_success_flag:

					print("Forward pass doomed")
					i = 2

					while not f_pass_success_flag:
 
						#print("Forward pass-trying %{}th time".format(i))
						self.alpha = self.alpha*0.9	#simulated annealing
						i += 1
						f_pass_success_flag = self.forward_pass(del_J_alpha)


			else:

				self.regularization_inc_mu()
				print("This iteration %{} is doomed".format(j))

			if j<5:
				self.alpha = self.alpha*.87
			else:
				self.alpha = self.alpha


		print(self.k)
		print(self.U_p)
		print(self.calculate_total_cost(self.X_0, self.X_p, self.U_p, self.N))
		



	def define_partial_func(self):

		self.f_x = Function('f_x' , [self.x, self.u] , [jacobian(self.dynamics_propagation_d(self.x, self.u) , self.x)])
		self.f_u = Function( 'f_u' , [self.x, self.u] , [jacobian(self.dynamics_propagation_d(self.x, self.u), self.u)])
		self.f_xx = Function( 'f_xx' , [self.x, self.u] , [jacobian(jacobian(self.dynamics_propagation_d(self.x, self.u) , self.x), self.x)])
		self.f_ux = Function( 'f_ux' , [self.x, self.u] , [jacobian(jacobian(self.dynamics_propagation_d(self.x, self.u) , self.x), self.u)])


		self.l_x = Function('l_x', [self.x, self.u], [gradient(self.cost(self.x, self.u) , self.x)])
		self.l_x_f = Function('l_x_f', [self.x], [gradient(self.cost_final(self.x) , self.x)])

		self.l_xx = Function('l_xx', [self.x, self.u], [jacobian(jacobian(self.cost(self.x, self.u) , self.x), self.x)])
		self.l_xx_f = Function('l_xx_f', [self.x], [jacobian(jacobian(self.cost_final(self.x) , self.x), self.x)])

		self.l_u = Function('l_u', [self.x, self.u], [gradient(self.cost(self.x, self.u), self.u)])
		self.l_uu = Function('l_uu', [self.x, self.u], [jacobian(jacobian(self.cost(self.x, self.u), self.u), self.u)])
		self.l_ux = Function('l_ux', [self.x, self.u], [jacobian(jacobian(self.cost(self.x, self.u), self.u), self.x)])



	def backward_pass(self):

		k_temp = copy.deepcopy(self.k)
		K_temp = copy.deepcopy(self.K)
		V_x_temp = copy.deepcopy(self.V_x)
		V_xx_temp = copy.deepcopy(self.V_xx)

		self.V_x[self.N-1] = self.l_x_f(self.X_p[self.N-1])	#X_p[-1]

		self.V_xx[self.N-1] = self.l_xx_f(self.X_p[self.N-1])	#X_p[-1]

		#initialize before forward pass
		del_J_alpha = 0

		for t in range(self.N-1, -1, -1):
			
			if t>0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.partials_list(self.X_p[t-1], self.U_p[t], self.V_x[t], self.V_xx[t])

			elif t==0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.partials_list(self.X_p_0, self.U_p[0], self.V_x[0], self.V_xx[0])

			if np.all(np.linalg.eigvals(Q_uu) <= 0):

				print("FAILED! Q_uu is not Positive definite at t=",t)
				#print(Q_uu, self.X_p[t-1], self.U_p[t])
				b_pass_success_flag = 0

				self.k = copy.deepcopy(k_temp)
				self.K = copy.deepcopy(K_temp)
				self.V_x = copy.deepcopy(V_x_temp)
				self.V_xx = copy.deepcopy(V_xx_temp)

				break

			else:

				b_pass_success_flag = 1
				# control-limited as follows
				self.k[t] = -mtimes(inv(Q_uu), Q_u) #self.clamp(-mtimes(inv(Q_uu), Q_u) , self.u_lb, self.u_ub) - self.U_p[t]
				self.K[t] = -mtimes(inv(Q_uu), Q_ux)
				'''
				if (abs(self.U_p[t][0] - self.u_lb[0]) < 1e-07) or (abs(self.U_p[t][0] - self.u_ub[0]) < 1e-07):
					
					#self.K[t][]
					self.K[t][0,2,4,6] = 0
					
					#print(self.K[t][0:4],  Q_ux)
				if (abs(self.U_p[t][1] - self.u_lb[1]) < 1e-07) or (abs(self.U_p[t][1] - self.u_ub[1]) < 1e-07):
					self.K[t][1,3,5,7] = 0
				
				if (np.linalg.norm(self.U_p[t] - self.u_lb) < 1e-07) or (np.linalg.norm(self.U_p[t] - self.u_ub) < 1e-07):

					self.K[t] = DM([[0,0,0,0],[0,0,0,0]])

				'''
				del_J_alpha += -self.alpha*mtimes(self.k[t].T, Q_u) - 0.5*self.alpha**2 * mtimes(self.k[t].T, mtimes(Q_uu, self.k[t]))

				if t>0:
					self.V_x[t-1] = Q_x + mtimes(self.K[t].T, mtimes(Q_uu, self.k[t])) + mtimes(self.K[t].T, Q_u) + mtimes(Q_ux.T, self.k[t])
					self.V_xx[t-1] = Q_xx + mtimes(self.K[t].T, mtimes(Q_uu, self.K[t])) + mtimes(self.K[t].T, Q_ux) + mtimes(Q_ux.T, self.K[t])

		self.count += 1
		return b_pass_success_flag, del_J_alpha



	def forward_pass(self, del_J_alpha):

		# Cost before forward pass
		J_1 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)

		self.X_p_temp = copy.deepcopy(self.X_p)
		self.U_p_temp = copy.deepcopy(self.U_p)

		for t in range(0, self.N):

			if t == 0:

				self.U_p[t] =self.U_p_temp[t] + self.alpha*self.k[t]
								 
				self.X_p[t] = self.dynamics_propagation_d(self.X_p_0, self.U_p[t])

			else:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] + mtimes(self.K[t], (self.X_p[t-1] - self.X_p_temp[t-1])) 

				self.X_p[t] = self.dynamics_propagation_d(self.X_p[t-1], self.U_p[t])

			#print(self.U_p[t],self.k[t],self.K[t], t)

		# Cost after forward pass
		J_2 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)

		z = (J_1 - J_2 )/del_J_alpha

		if z < self.c_1:

			self.X_p = copy.deepcopy(self.X_p_temp)
			self.U_p = copy.deepcopy(self.U_p_temp)
			f_pass_success_flag = 0
			print("f",z, del_J_alpha, J_1, J_2)

		else:

			f_pass_success_flag = 1
			#print("successful forward pass with cost jumping from",J_1,"to", J_2)
			#print(self.k, self.U_p)

		return f_pass_success_flag



	def partials_list(self, x, u, V_x_next, V_xx_next):


		#F_xx = self.f_xx(x, u)
		#F_ux = self.f_ux(x, u)

		F_x = self.f_x(x, u)
		F_u = self.f_u(x, u)

		Q_x = self.l_x(x, u) + mtimes(F_x.T, V_x_next)
		Q_u = self.l_u(x, u) + mtimes(F_u.T, V_x_next)
		'''
		2nd order dynamics
		Fxx = 0
		Fux = 0
		
		for i in range(4):
			Fxx += V_x_next[i]*np.reshape(F_xx, (4,4,4))[:,i,:]
			Fux += V_x_next[i]*np.reshape(F_ux, (4,4,2))[i,:,:]
		'''
		Q_xx = self.l_xx(x, u) + mtimes(F_x.T, mtimes(V_xx_next, F_x)) #+ Fxx#np.tensordot(V_x_next.T, np.reshape(F_xx, (4,4,4)), axes=(1,1))[0]#tensor_1
		Q_ux = self.l_ux(x, u) + mtimes(F_u.T, mtimes(V_xx_next + self.mu*np.identity(V_xx_next.shape[0]), F_x)) #+ Fux.T#np.tensordot(V_x_next.T, np.reshape(F_ux, (2,4,4)), axes=(1,2))[0]#tensor_2
		Q_uu = self.l_uu(x, u) + mtimes(F_u.T, mtimes(V_xx_next + self.mu*np.identity(V_xx_next.shape[0]), F_u)) #+ mtimes(V_x_next, self.f_uu(x, u)) -> this term is zero because of affine assumption

		return Q_x, Q_u, Q_xx, Q_uu, Q_ux

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

	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		#initialize total cost
		cost_total = self.cost(initial_state, control_traj[0])

		for t in range(0, horizon-1):

			cost_total += self.cost(state_traj[t], control_traj[t+1])

		cost_total += self.cost_final(state_traj[horizon-1])

		return cost_total

	def regularization_inc_mu(self):

		# increase mu - regularization 

		self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)

		self.mu = np.maximum(self.mu_min, self.mu*self.delta)

		if self.mu > self.mu_max:
			self.mu = self.mu_max


		print(self.mu)

	def regularization_dec_mu(self):

		# decrease mu - regularization 

		self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)

		if self.mu*self.delta > self.mu_min:
			self.mu = self.mu*self.delta

		else:
			self.mu = self.mu_min

	def boxQP(self, H, g, u_lb, u_ub, x0):
		# solves the problem :- Minimize 0.5*x'*H*x + x'*g  s.t. u_lb<=x<=u_ub
		a = time.time()
		x = self.clamp(x0, u_lb, u_ub)
		max_iter = 50
		clamped = np.zeros(x.shape)
		nfactor = 0
		minGrad = 1e-6
		Armijo = 0.1
		stepDec = 0.6

		#intitial cost of the objective function
		value = mtimes(x.T, g) + 0.5*mtimes(x.T, mtimes(H, x))
		xc = copy.deepcopy(x)
		#print(x0, x, xc)
		for iter in range(0, max_iter):

			oldvalue = value

			#find gradient
			grad = g + mtimes(H, x)

			# find clamped dimensions
			clamped_old = clamped
			clamped = np.zeros(x.shape)
			
			# list for storing x as x_free and x_c variables
			x_f, x_c, g_f, g_c, H_ff, H_fc = [[] for i in range(6)]

			for i in range(0, x.shape[0]):

				if (x[i][0] == u_lb[i]) and (grad[i]>0):
					clamped[i] = 1
					x_c = np.append(x_c, x[i][0])
					g_c = np.append(g_c, g[i][0])

				elif (x[i][0] == u_ub[i]) and (grad[i]<0):

					clamped[i] = 1
					x_c = np.append(x_c, x[i][0])
					g_c = np.append(g_c, g[i][0])

				else:

					x_f = np.append(x_f, x[i][0])
					g_f = np.append(g_f, g[i][0])

			# negate
			free = 1 - clamped

			if all(clamped):
				break
			elif free[0][0] and not free[1][0]:

				H_ff = H[0]
				H_fc = H[1]

			elif free[1][0] and not free[0][0]:

				H_ff = H[3]
				H_fc = H[2]

			if free[0][0] and free[1][0]:

				H_ff = H
				H_fc = []

		

			if np.linalg.norm(grad) < minGrad:
				break

			#get search direction
			#search = np.zeros((len(x), 1))
			del_x = []

			if len(x_c)==0:
				
				del_x_f = -mtimes(inv(H_ff), DM(g_f) ) - DM(x_f)
				del_x = np.append(del_x, del_x_f)
			else:

				del_x_f = np.array(-mtimes(inv(H_ff), (g_f + mtimes(H_fc, x_c[0]))) - x_f)
				del_x = np.append(del_x, del_x_f)
				del_x = np.append(del_x, np.zeros(len(x_c)))

			step = 1
			nstep = 0
				
			xc = self.clamp(x + step*del_x, u_lb, u_ub)
			vc = mtimes(xc.T, g) + 0.5*mtimes(xc.T, mtimes(H, xc))
			sdotg = (mtimes(g.T, x - xc))
			
			if not np.all(np.array(del_x) < .001*np.ones(x.shape)):# and np.array(x) == np.zeros(x.shape))	:

				while ( oldvalue - vc)/mtimes(g.T, x - xc) < Armijo:
				
					#print("h", iter, del_x, x, C_old, C_1, step, ( C_old - C_1)/mtimes(g.T, x - xc))
					step = step * stepDec
					nstep += 1
					xc = self.clamp(x + step*del_x, u_lb, u_ub)		
					vc = mtimes(xc.T, g) + 0.5*mtimes(xc.T, mtimes(H, xc))
					print(( oldvalue - vc)/mtimes(g.T, x - xc))
			x = xc
			value = vc

		return xc, H_ff, free
		
	def clamp(self, u, u_lb, u_ub):

		u_projected = fmin(fmax(u, u_lb), u_ub)

		return u_projected



