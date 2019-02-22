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
horizon = 229
C_Hz = 120 # Horizon for a window in MPC,control horizon

# Car parameters:

L = .335 # Length of the car
W = .3 # Width of the car


r_u = [.7, -.7]  # Bounds of linear velocity
r_w = [1.3, -1.3]     # Bounds of Angular Velocity 

# Optimization parameters

n_x = 4   # No. of state variables
n_u = 2   # No. of control variables

# Weight matrices in objective function
W_x = 	DM([[12,0,0,0],[0,12,0,0],[0,0,2.2,0],[0,0,0,2.2]])
W_u =   200*DM([[1, 0], [0, 1]])
W_x_f = 	500*DM([[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]])

#Obstcale function quotients
Q_1 = DM([[4.8106, .8590, 0, 0], [.859, 4.8106, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
Q_2 = DM([[12.50, 0, 0, 0], [0, .0108, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
Q_3 = DM([[6.25, 0, 0, 0], [0, 2.7778, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
Q_4 = DM([[12.5, 0, 0, 0], [0, .0108, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
Q_5 = DM([[1.3889, 0, 0, 0], [0, 1.3889, 0, 0], [0, 0, 0, 0], [0,0,0,0]])


#obstacle centers 
x_obstacle_1 = DM([0.8030, 1.303, 0, 0])
pho_1 = 15
x_obstacle_2 = DM([2.1, -3, 0, 0])
pho_2 = 15
x_obstacle_3 = DM([1.6, 4.4, 0, 0])
pho_3 = 40
x_obstacle_4 = DM([3.3, 7.8, 0, 0])
pho_4 = 15
x_obstacle_5 = DM([4.5, 3.5, 0, 0])
pho_5 = 15


#C_t = 2*vertcat(horzcat(W_x, DM.zeros(3, 2)), horzcat(DM.zeros(2, 3), W_u)) 

W_x_LQR = DM([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]])
W_x_LQR_f = 25*DM([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
W_u_LQR =  DM([[.5, 0],[0, 1]]) #00000000000000


x_g_proximity = [.05, .1]  # distance and absolute angle errors respectively

obstacle_1 = Polygon([(0.5, 1.250), (.750, 1.250), (.75, 1.0), (1.0, 1.0), (1.0, 1.50), (0.5, 1.5), (0.5, 1.250)])
obstacle_2 = Polygon([(2, 0), (2, 3.5), (2.2, 3.5), (2.2, 0), (2, 0)])
obstacle_3 = Polygon([(3.2, 1.50), (3.4, 1.5), (3.4, 5.0), (3.2, 5.0), (3.2, 1.50)])
obstacle_4 = Polygon([(1.2, 4.4), (1.6, 4.0), (2.0, 4.40), (1.6, 4.8), (1.2, 4.40)])
obstacle_5 = Polygon([(4.0, 1.0), (4.4, 1.0), (4.4, 0.8), (4.9, 0.8), (4.9, 0.1), (4.0, .1), (4.0, 1.0)])
obstacle_6 = Polygon([(4, 3), (4, 4), (5, 4), (5, 3), (4, 3)])
obstacle_7 = Polygon([(-1.0, -1.0), (6, -1), (6, 6), (-1, 6), (-1, -1), (-1.5, -1.5), (6.5, -1.5), (6.5, 6.5), (-1.5, 6.5), (-1.5, -1.5)])

