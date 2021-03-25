#!/usr/bin/env python

"""
GPs_learning.py

Learning GP models for human motion and intention.

Author: Roya Sabbagh Novin (sabbaghnovin@gmail.com)
"""

import numpy as np
from scipy import *
from Gaussian_processes import MGPR
import pickle

class motion_set():
    """
    A class to stor all motion data and models for a patient in a hospital room.
    """
    def __init__(self, senario_set):
        self.starts = senario_set['start']
        self.ends = senario_set['end']
        self.probs = [[0.8, 0.2],
                 [0.7, 0.1, 0.2],
                 [0.8, 0.2],
                 [0.05, 0.6, 0.1, 0.2, 0.05],
                 [0.3, 0.2, 0.5],
                 [0.2, 0.2, 0.6],]
        self.GPs = [[], [], [], [], [], []]


if __name__ == '__main__':

    # Load generated trajectory dataset
    design_name = "Room-2-Inboard-Footwall"
    path = "/home/roya/catkin_ws/src/risk_aware_planning"
    background_filename = "{0}/Room_Designs/{1}_objects_rotated.png".format(path, design_name)

    pickle_trajectories = open("data/{0}/trajectories.pickle".format(design_name),"rb")
    patient_motion = pickle.load(pickle_trajectories)
    pickle_trajectories.close()

    # Load previously learned models, comment if learning from scratch
    pickle_motion = open("data/{0}/motion_GPs.pickle".format(design_name),"rb")
    motion = pickle.load(pickle_motion)
    pickle_motion.close()

    # Learn GP Models
    for initial in range(len(patient_motion.scenario_set['start'])):
    # for initial in range(4,5): Use this line if only learning for a specific scenario
        motion.GPs[initial] = []
        for goal in range(len(patient_motion.scenario_set['end'][initial])):
            trajectories = patient_motion.scenario_set['trajs'][initial][goal]
            path = []
            dx = []
            print("initial:", initial, "goal:", goal)
            for step in range(len(trajectories[0][0])-1):
                for traj in trajectories:
                    path.append([traj[1][step],traj[2][step]])
                    dx.append([traj[1][step+1]-traj[1][step],traj[2][step+1]-traj[2][step]])

            data = [array(path).reshape(-1,2),array(dx).reshape(-1,2)]

            initials_x = [data[0][i][0] for i in range(20)]
            initials_y = [data[0][i][1] for i in range(20)]
            m = array([mean(initials_x),mean(initials_y)])
            s =  array([[std(initials_x), 0],
                  [0, std(initials_y)]]) #0.05 * np.eye(2)
            n = 20
            model = MGPR(data)
            model.optimize(restarts=5)

            params_set = model.save_model()
            print(params_set)
            motion.GPs[initial].append(params_set)

    pickle_motion = open("data/{0}/motion_GPs.pickle".format(design_name),"wb")
    pickle.dump(motion, pickle_motion)
    pickle_motion.close()
