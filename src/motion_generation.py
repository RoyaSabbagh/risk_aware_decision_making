#!/usr/bin/env python

import numpy as np
from utility import define_obstacles
import pickle
from Trajectory_generation import motion_generator


if __name__ == '__main__':

    design_name = "Room-2-Inboard-Footwall"
    path = "/home/roya/catkin_ws/src/risk_aware_planning"
    background_filename = "{0}/Room_Designs/{1}_objects_rotated.png".format(path, design_name)

    pickle_env = open("data/{0}/env.pickle".format(design_name),"rb")
    env = pickle.load(pickle_env)
    pickle_env.close()

    obstacles = define_obstacles(env) # Defines obstacles including furniture and walls

    patient_motion = motion_generator(env, obstacles, num_points=20, dt=0.5, scenario_set={'start': ['Main Door','Bed_L', 'Bed_R', 'Toilet', 'Chair-Visitor'],
                                                                                            'end': [['Bed_R', 'Toilet'],
                                                                                                    ['Toilet', 'Main Door'],
                                                                                                    ['Toilet', 'Main Door'],
                                                                                                    ['Chair-Visitor', 'Main Door', 'Bed_L', 'Bed_R'],
                                                                                                    ['Toilet', 'Main Door', 'Bed_L']],
                                                                                            'trajs':[[], [], [], [], [], []],
                                                                                            }, num_trial=20, v=[0.6, 0.2], w=[0.6, 0.2])
    patient_motion.generate_trajectories()
    patient_motion.plot(background_filename)

    pickle_trajectories = open("data/{0}/trajectories.pickle".format(design_name),"wb")
    pickle.dump(patient_motion, pickle_trajectories)
    pickle_trajectories.close()
