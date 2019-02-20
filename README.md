# T_PFC_paper
Trajectory optimized perturbation feedback controller

Libraries needed:
- Casadi (for symbolic differentiation and optimization- also comes with 'ipopt' optimization library)
- h5py
- ROS

Implementations:
- T-PFC
- DDP
- T-LQR
- MPC

Use:
To run the examples, set your $PYTHONPATH to T_PFC_paper folder. If you are using command line,it can be done through 'export PYTHONPATH=$PYTHONPATH:/YOUR_FOLDER'.

Examples:
- Car-like robot
- Car-like robot with trailers attached
- Manipulator
