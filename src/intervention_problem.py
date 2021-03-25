#!/usr/bin/env python

"""
intervention_problem.py

Defining the intervention problem.

Author: Roya Sabbagh Novin (sabbaghnovin@gmail.com)
"""

import math
import numpy as np
from shapely.geometry import Polygon, Point


class intervention_problem():
    """
    A class to define the intervention problem, including costs and collision check.
    """
    def __init__(self, fall_risk, env, robot, patient, pred_trajs=None, alpha = 0.7, beta=10, rho=0.001):

        self.pred_trajs = pred_trajs
        self.fall_risk = fall_risk
        self.env = env
        self.beta = beta
        self.rho = rho
        self.robot = robot
        self.patient = patient
        self.alpha = alpha
        self.dt = 0.5

    def update(self, pred_trajs, robot):

        self.pred_trajs = pred_trajs
        self.robot = robot

    def cost_exp_CVaR(self, intervention):

        return self.cost_exp(intervention) + self.beta*self.cost_CVaR(intervention)

    def cost_exp(self, intervention):

        if intervention[2]< 0.5:
            return float("inf")
        collision = self.check_collision(intervention)
        if collision:
            return float("inf")
        cost_p = self.cost_patient(intervention, "EXP")
        cost_R = self.cost_robot(intervention)
        cost = cost_p + self.rho * cost_R
        return cost

    def cost_CVaR(self, intervention):

        if intervention[2]< 0.5:
            return float("inf")
        collision = self.check_collision(intervention)
        if collision:
            return float("inf")
        cost_p = self.cost_patient(intervention, "CVaR")
        cost_R = self.cost_robot(intervention)
        cost = cost_p + self.rho * cost_R
        return cost

    def cost_robot(self, intervention):

        cost, _ = self.robot.manipulation_plan(intervention)
        return cost

    def cost_patient(self, intervention, cost_type):

        costs = []
        if cost_type == "EXP":
            for traj in self.pred_trajs:
                costs.append(sum(self.cost_traj(traj, intervention))/(len(traj)-1)*traj[-1])
            cost = sum(costs)/len(costs)
        elif cost_type == "CVaR":
            for traj in self.pred_trajs:
                cost_t = self.cost_traj(traj, intervention)
                for i in range(len(cost_t)):
                    costs.append(cost_t[i]*traj[-1])
            costs.sort()
            n_alpha = int(len(cost_t)*self.alpha)
            cost = sum(costs[n_alpha:-1])/(len(costs)-n_alpha)
        return cost

    def cost_traj(self, traj, intervention):

        index = self.find_intervention(traj, intervention)
        n = len(traj)-1
        if index == -1:
            _ , cost = self.fall_risk.getDistibutionForTrajectory(traj[0:n], False, False)
        else:
            _ , cost_before = self.fall_risk.getDistibutionForTrajectory(traj[0:index], False, False)
            _ , cost_after = self.fall_risk.getDistibutionForTrajectory(traj[index:n], False, True)
            cost = cost_before + cost_after
        return cost

    def find_intervention(self, traj, intervention):

        for i in range(len(traj)-1):
            if ((traj[i][0]-intervention[0])**2 + (traj[i][1]-intervention[1])**2 <= 0.4) and (self.dt*i>=intervention[2]):
                return i
        return -1

    def cost_deterministic(self, traj, index):

        n = len(traj)-1
        _ , cost_before = self.fall_risk.getDistibutionForTrajectory(traj[0:index], False, False)
        _ , cost_after = self.fall_risk.getDistibutionForTrajectory(traj[index:n], False, True)
        cost_p = sum(cost_before + cost_after)
        intervention = [traj[index][0], traj[index][1], index*self.dt]
        cost_R = self.cost_robot(intervention)
        cost = cost_p + self.rho * cost_R
        return cost

    def check_collision(self, intervention):

        if intervention[0]<3.5 or intervention[0]>6.3:
            return 1
        if intervention[1]<2 or intervention[1]>6.3:
            return 1
        point = Point([intervention[0],intervention[1]])
        for obj in self.env.objects:
            if point.within(obj.polygon):
                return 1
        return 0
