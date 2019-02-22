# List of Parameters for T-LQR for car-like robot
# This file contains all the parameters that the other python code files use.
# Date: 6/19/18 
# Copyright @ P. Karthikeya Sharma

from math import *
import numpy as np
from casadi import *
from shapely.geometry.polygon import Polygon

# Time-step (in sec)

dt = 0.10

'''
num_lines = sum(1 for line in open('/home/karthikeya/catkin_ws/src/T-LQG_car/k_lqg/python_implementation/rrt_path.txt'))
K = num_lines # Total horizon
'''
#N = 20 # Horizon for a window in MPC
horizon = 180
# Car parameters:

L = .5 # Length of the car
W = .3 # Width of the car

l_cr = {0:1, 1:1, 2:1, 3:1} # lengths of connecting rods
d_1 = 1 # length of the connecting rod
L_1 = .5 # Length of the trailer
W_1 = .3 # width of the trailer

d_2 = 1 # length of the connecting rod
L_2 = .5 # Length of the trailer
W_2 = .3 # width of the trailer

d_3 = 1 # length of the connecting rod
L_3 = .5 # Length of the trailer
W_3 = .3 # width of the trailer



r_u = [.7, -0.7]  # Bounds of linear velocity
r_w = [1.3, -1.3]     # Bounds of Angular Velocity 

theta_2_bounds = [1.309, -1.309]

# Optimization parameters
n_x = 6    # No. of state variables
n_u = 2    # No. of control variables


# Weight matrices in objective function
W_x = 	10*DM([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1.5, 0, 0, 0], [0, 0, 0, 1.5, 0, 0], [0, 0, 0,0, 3, 0], [0, 0, 0, 0, 0, 3]])
W_u =   50*DM([[7, 0], [0, 5]])#DM([[100000, 0],[0, 2]]) #00000000000000

n_obstacles = 1
#Obstcale function quotients
Q_obs = {
0 : DM([[12.5, 0, 0, 0, 0, 0], [0, 0.125, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
1 : DM([[2, 0, 0, 0, 0, 0], [0, .2222, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),			#.0108 .0397
}

#obstacle centers 
x_obstacle = { 0: DM([1.2, -.5, 0, 0, 0, 0]), 1: DM([2.5, 3.5, 0, 0, 0, 0]) }
pho = {0:25, 1:40, 2:40}


#C_t = 2*vertcat(horzcat(W_x, DM.zeros(3, 2)), horzcat(DM.zeros(2, 3), W_u)) 

W_x_LQR = 10*DM([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
W_x_LQR_f = 25*DM([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
W_u_LQR =  DM([[.5, 0],[0, 1]]) #00000000000000

W_x_f = 	1000*DM([[100,0,0,0,0,0],[0,100,0,0,0,0],[0,0,100,0,0,0],[0,0,0,100,0,0],[0,0,0,0,150,0],[0,0,0,0,0,150]])

x_g_proximity = [.05, .1]  # distance and absolute angle errors respectively

obstacle_1 = Polygon([(1, -3), (1, 1.50), (1.2, 1.50), (1.20, -3.0), (1.0, -3.0)])
obstacle_2 = Polygon([(1.3, 0), (1.3, 2), (1.7, 2), (1.7, 0), (1.3, 0)])
obstacle_3 = Polygon([(1.2, 4.4), (1.6, 4.0), (2.0, 4.40), (1.6, 4.8), (1.2, 4.40)])
obstacle_4 = Polygon([(4.0, 1.0), (4.4, 1.0), (4.4, 0.8), (4.9, 0.8), (4.9, 0.1), (4.0, .1), (4.0, 1.0)])

obstacle_5 = Polygon([(-1, -.5), (6, -.5), (6, 5.5), (-1, 5.5), (-1, -.5)])