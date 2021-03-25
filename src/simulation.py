#!/usr/bin/env python


import math
import numpy as np
from shapely.geometry import Point
from scipy import *
from utility import is_near_sitting_object, define_obstacles
import random
from Gaussian_processes import MGPR
import gpflow
import rospy
from copy import deepcopy
from Optimization import OptPath



class sim_patient():

    def __init__(self, motion, trajs, env, fall_risk):

        self.motion = motion
        self.trajs = trajs
        self.env = env
        self.fall_risk = fall_risk
        self.sampled_traj = []
        self.initial = None
        self.goal = None
        self.models = []

        self.motion.probs = [[0.8, 0.2],
                 [0.7, 0.1, 0.2],
                 [0.3, 0.7],
                 [0.2, 0.4, 0.05, 0.2, 0.1],
                 [0.3, 0.2, 0.5],
                 [0.2, 0.2, 0.6],]

    def sample_traj(self):

        # self.initial = random.randint(0, len(self.motion.starts)-2)
        self.initial = 1
        # print(self.motion.ends[self.initial])
        # eps=np.random.uniform(0,1)
        # lb=0
        # ub=0
        # for p in range(len(self.motion.ends[self.initial])):
        #     ub+=self.motion.probs[self.initial][p]
        #     if (lb<=eps and eps<ub):
        #         self.goal = p
        #         break
        #     lb=ub

        self.goal = 0# random.choice([0,2,3,4])
        print(self.trajs.scenario_set['end'][self.initial][self.goal])
        # rospy.loginfo(self.trajs.scenario_set['start'][self.initial])
        # rospy.loginfo(self.trajs.scenario_set['end'][self.initial][self.goal])

        trajectory = self.trajs.traj_plan(self.initial,self.goal)
        path = []

        # rospy.loginfo(trajectory)
        last = [-1,-1]
        for step in range(len(trajectory[0])-1):
            if (sqrt((last[0]-trajectory[0][step][0])**2+ (last[1]-trajectory[0][step][1])**2)>0.1) or (trajectory[0][step][5]!='sit_to_stand') or (trajectory[0][step][5]!='stand_to_sit'):
                path.append([array([trajectory[0][step][0],trajectory[0][step][1], trajectory[0][step][2], trajectory[0][step][3], trajectory[0][step][4], trajectory[0][step][5]]), trajectory[0][step][6]])
                last = [trajectory[0][step][0],trajectory[0][step][1]]
        self.sampled_traj = array(path)
        # rospy.loginfo(self.sampled_traj)

        for goal in range(len(self.motion.ends[self.initial])):
            trajectories = self.trajs.scenario_set['trajs'][self.initial][goal]
            path = []
            dx = []
            for step in range(len(trajectories[0][0])-1):
                for t in trajectories:
                    path.append([t[1][step],t[2][step]])
                    dx.append([t[1][step+1]-t[1][step],t[2][step+1]-t[2][step]])

            data = [array(path).reshape(-1,2),array(dx).reshape(-1,2)]
            self.models.append(MGPR(data))

            i=0
            n=10
            for m in self.models[goal].models:
                gpflow.utilities.multiple_assign(m, self.motion.GPs[self.initial][goal][i])
                i+=1

    def predict_motion(self, observed_traj, n_traj = 5):

        goal_probabilities = []
        pred_trajs = []

        for goal in range(len(self.motion.ends[self.initial])):
            n=10
            goal_probability = self.models[goal].predict_intention_probability(observed_traj, self.motion.probs[self.initial][goal])
            goal_probabilities.append(deepcopy(goal_probability))

        sum_p = sum([goal_probabilities[j] for j in range(len(goal_probabilities))])
        for j in range(len(goal_probabilities)):
            goal_probabilities[j] = goal_probabilities[j]/sum_p

        for goal in range(len(self.motion.ends[self.initial])):
            for i in range(n_traj):
                pred_traj = self.models[goal].predict_path(np.array(observed_traj[-1]), n)
                tau = []
                for i in range(len(pred_traj[0])):
                    tau.append([pred_traj[0][i][0], pred_traj[0][i][1], math.atan2(pred_traj[1][i][1], pred_traj[1][i][0]), pred_traj[1][i][0], pred_traj[1][i][1], 'walking'])
                if is_near_sitting_object(Point(tau[0]), self.env, self.trajs.scenario_set['start'][self.initial]) :
                    tau[0][5] = 'sit_to_stand'
                elif is_near_sitting_object(Point(tau[-1]), self.env, self.trajs.scenario_set['end'][self.initial][self.goal]):
                    tau[-1][5] = 'stand_to_sit'
                tau.append(deepcopy(goal_probabilities[goal]))
                pred_trajs.append(tau)
            # Evaluating the predicted trajectory
            # pred_trajs.append(self.fall_risk.getDistibutionForTrajectory(tau, False, False))


        return goal_probabilities, pred_trajs


class sim_robot():

    def __init__(self, env, robot_state, object_state, dt=1):

        self.env = env
        self.object_state = object_state
        self.robot_state = robot_state
        self.offsets_robot = [self.robot_state.point.x - self.object_state.point.x, self.robot_state.point.y - self.object_state.point.y, self.robot_state.point.z - self.object_state.point.z]
        self.dt = dt

    def manipulation_plan(self, intervention):
        obstacles = define_obstacles(self.env)
        cost, path = OptPath([self.object_state.point.x, self.object_state.point.y, self.object_state.point.z], [intervention[0],intervention[1]], obstacles, intervention[2])

        return cost, path

    def execute(self, plan):
        # print("plan:", plan)
        self.update_object(plan[1], False)
        self.robot_state.point.x = self.object_state.point.x + self.offsets_robot[0]
        self.robot_state.point.y = self.object_state.point.y + self.offsets_robot[1]
        self.robot_state.point.z = self.object_state.point.z + self.offsets_robot[2]

    def update_object(self, new_state, assisted):
        if assisted:
            self.object_state.point.x = new_state[0]
            self.object_state.point.y = new_state[1]#+0.2*np.sin(new_state[3])
            self.object_state.point.z = new_state[3]
        else :
            self.object_state.point.x = new_state[0]
            self.object_state.point.y = new_state[1]
            # self.object_state.point.z = new_state[2]
