#!/usr/bin/env python


from Optimization import OptPath_patient
import math
import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon, Point
from scipy import *
from utility import define_obstacles, sample_point, is_near_sitting_object, RBF, find_corners
import random
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.cbook as cbook
import matplotlib.image as image
from scipy.interpolate import interp1d



class motion_generator():

    def __init__(self, env, obstacles, num_points=20, dt=0.5, scenario_set={'start': ['Main Door'], 'end': ['Toilet']}, num_trial=20, v=[0.6, 0.2], w=[0.6, 0.2]):
        self.num_trial = num_trial
        self.scenario_set = scenario_set
        self.dt = dt
        self.num_points = num_points
        self.v = v # maximum linear velocity = [mu, sigma]
        self.w = w # maximum angular velocity = [mu, sigma]
        self.obstacles = obstacles
        self.env = env

    def traj_plan(self, initial, goal):
        ''' This is the main function that generates trajectories given a scenario. It samples the start and end point.
        Then, using optimization, we find an optimal path between these 2 points. Finally, for each point in the trajectory,
        we find a corresponding activity based on the distance to the target object. '''

        found = 0
        while found == 0:
            # Sample points near the start and end locations
            v_max = random.gauss(self.v[0], self.v[1])
            w_max = random.gauss(self.w[0], self.w[1])
            if self.scenario_set['start'][initial] == "Toilet":
                phi0 = 1.57
                x0 = 4.75
                y0 = 2.5
            elif self.scenario_set['start'][initial] == "Bed_L":
                phi0 = 1.57
                x0 = 5.5
                y0 = 5.6
            elif self.scenario_set['start'][initial] == "Bed_R":
                phi0 = -1.57
                x0 = 5.5
                y0 = 4.5
            elif self.scenario_set['start'][initial] == "Chair-Patient":
                phi0 = -2.2
                x0 = 5.8
                y0 = 4.5
            elif self.scenario_set['start'][initial] == "Chair-Visitor":
                phi0 = -1.57
                x0 = 5
                y0 = 6

            if self.scenario_set['end'][initial][goal] == "Toilet":
                phi_f = 1.57
                x_f = 4.75
                y_f = 2.8
            elif self.scenario_set['end'][initial][goal] == "Bed_L":
                phi_f = 1.57
                x_f = 5.5
                y_f = 5.6
            elif self.scenario_set['end'][initial][goal] == "Bed_R":
                phi_f = -1.57
                x_f = 5.5
                y_f = 4.5
            elif self.scenario_set['end'][initial][goal] == "Chair-Patient":
                phi_f = -2.2
                x_f = 5.8
                y_f = 4.5
            elif self.scenario_set['end'][initial][goal] == "Chair-Visitor":
                phi_f = -1.57
                x_f = 5
                y_f = 5.5

            patient_s = [x0, y0, phi0, 0, 0]
            patient_g = [x_f, y_f, phi_f, 0, 0]
            print("start,goal:", patient_s, patient_g)
            # patient_s = sample_point(self.env, self.scenario_set['start'][initial], self.obstacles)
            # patient_g = sample_point(self.env, self.scenario_set['end'][initial][goal], self.obstacles)
            traj_init = {'start': patient_s, 'end': patient_g, 'v_max': v_max, 'w_max': w_max}
            # print("senario: ", scenario)
            # Find a trajectory between sampled points
            T, cost, tau, status = OptPath_patient(traj_init['start'], traj_init['end'], [traj_init['v_max'], traj_init['w_max']] , self.obstacles, self.num_points, assistive_device=False)
            # If the optimization was successful, find the type of activity for each point on the trajectory and add it to the resturning path
            if status == 2 :
                found = 1
                print("tau:", tau)
                sit_to_stand = False
                stand_to_sit = False
                if is_near_sitting_object(Point(tau[0]), self.env, self.scenario_set['start'][initial]) :

                    if self.scenario_set['start'][initial] == "Toilet":
                        phi0 = 1.57
                        x0 = 4.75
                        y0 = 2.3
                    elif self.scenario_set['start'][initial] == "Bed_L":
                        phi0 = 1.57
                        x0 = 5.5
                        y0 = 5.4
                    elif self.scenario_set['start'][initial] == "Bed_R":
                        phi0 = -1.57
                        x0 = 5.5
                        y0 = 4.8
                    elif self.scenario_set['start'][initial] == "Chair-Patient":
                        phi0 = -2.2
                        x0 = 5.8
                        y0 = 4.5
                    elif self.scenario_set['start'][initial] == "Chair-Visitor":
                        phi0 = -1.57
                        x0 = 5
                        y0 = 6.35

                    if phi0 > np.pi:
                        phi0-=2*np.pi
                    elif phi0 < -np.pi:
                        phi0+=2*np.pi

                    sit_to_stand = True

                if is_near_sitting_object(Point(tau[-1]), self.env, self.scenario_set['end'][initial][goal]):

                    if self.scenario_set['end'][initial][goal] == "Toilet":
                        phi_f = 1.57
                        x_f = 4.65
                        y_f = 2.55
                    elif self.scenario_set['end'][initial][goal] == "Bed_L":
                        phi_f = 1.57
                        x_f = 5.5
                        y_f = 5.4
                    elif self.scenario_set['end'][initial][goal] == "Bed_R":
                        phi_f = -1.57
                        x_f = 5.5
                        y_f = 4.8
                    elif self.scenario_set['end'][initial][goal] == "Chair-Patient":
                        phi_f = -2.2
                        x_f = 5.8
                        y_f = 4.5
                    elif self.scenario_set['end'][initial][goal] == "Chair-Visitor":
                        phi_f = -1.57
                        x_f = 5
                        y_f = 6

                    if phi_f > np.pi:
                        phi_f-=2*np.pi
                    elif phi_f < -np.pi:
                        phi_f+=2*np.pi

                    stand_to_sit = True

                if sit_to_stand:
                    phi_start = math.atan2(tau[0][0]-x0, tau[0][1]-y0) +np.pi/2
                    if phi_start > np.pi:
                        phi_start-=2*np.pi
                    elif phi_start < -np.pi:
                        phi_start+=2*np.pi

                    tau_new = deepcopy(tau[0])
                    tau_new[0] = x0
                    tau_new[1] = y0
                    tau_new[3] = phi_start
                    tau.insert(0, deepcopy(tau_new))

                if stand_to_sit:
                    phi_finish = math.atan2(tau[-1][0]-x_f, tau[-1][1]-y_f) +np.pi/2
                    if phi_finish > np.pi:
                        phi_finish-=2*np.pi
                    elif phi_finish < -np.pi:
                        phi_finish+=2*np.pi

                    tau_new = deepcopy(tau[-1])
                    tau_new[0] = x_f
                    tau_new[1] = y_f
                    tau_new[3] = phi_finish
                    tau[-1][3] = phi_finish
                    tau.append(deepcopy(tau_new))

                nn = len(tau)
                for i in range(0, nn-1):
                    n_p = 20
                    tau[((n_p+1)*i)].append('walking')
                    dphi = tau[((n_p+1)*i)+1][3]-tau[((n_p+1)*i)][3]
                    if abs(dphi)> np.pi:
                        dphi_sign = dphi/abs(dphi);
                        dphi = dphi_sign*(abs(dphi)-(2*np.pi));

                    xnew = np.linspace(tau[((n_p+1)*i)+1][0], tau[((n_p+1)*i)][0], num=n_p, endpoint=True)
                    ynew = np.linspace(tau[((n_p+1)*i)+1][1], tau[((n_p+1)*i)][1], num=n_p, endpoint=True)
                    phinew = np.linspace(tau[((n_p+1)*i)][3]+dphi, tau[((n_p+1)*i)][3], num=n_p, endpoint=True)

                    tau_new = deepcopy(tau[((n_p+1)*i)])

                    for j in range(n_p):
                        tau_new[0] = xnew[j]
                        tau_new[1] = ynew[j]
                        tau_new[3] = phinew[j]
                        tau.insert(((n_p+1)*i)+1, deepcopy(tau_new))

                tau[-1].append('walking')

                if sit_to_stand:

                    tau_new = deepcopy(tau[0])
                    tau_new[0] = x0
                    tau_new[1] = y0
                    tau_new[3] = phi0
                    tau_new[4] = 0
                    tau_new[5] = 0
                    tau_new[6]='sit_to_stand'

                    x = np.linspace(0, 10, num=11, endpoint=True)
                    y = np.array([0.56, 0.54, 0.52, 0.5, 0.52, 0.5, 0.52, 0.62, 0.64, 0.7, 0.75])
                    f = interp1d(x, y, kind='cubic')

                    xnew = np.linspace(0, 10, num=50, endpoint=True)
                    phinew = np.linspace(phi_start, tau_new[3], num=50, endpoint=True)
                    ynew = f(xnew)
                    ynew = ynew[::-1]
                    for j in range(50):
                        tau_new[2] = ynew[j]
                        tau_new[3] = phinew[j]
                        tau.insert(0, deepcopy(tau_new))

                if stand_to_sit:
                    phinew = np.linspace(tau[-1][3], phi_f, num=50, endpoint=True)
                    tau_new = deepcopy(tau[-1])
                    for j in range(50):
                        tau_new[3] = phinew[j]
                        tau.append(deepcopy(tau_new))

                    tau_new = deepcopy(tau[-1])
                    tau_new[0] = x_f
                    tau_new[1] = y_f
                    tau_new[3] = phi_f
                    tau_new[4] = 0
                    tau_new[5] = 0
                    tau_new[6]='stand_to_sit'

                    x = np.linspace(0, 10, num=11, endpoint=True)
                    y = np.array([0.56, 0.54, 0.52, 0.5, 0.52, 0.5, 0.52, 0.62, 0.64, 0.7, 0.75])
                    f = interp1d(x, y, kind='cubic')

                    xnew = np.linspace(0, 10, num=50, endpoint=True)
                    ynew = f(xnew)[::-1]
                    for j in range(50):
                        tau_new[2] = ynew[j]
                        tau.append(deepcopy(tau_new))


        return [tau, patient_s, patient_g, T]

    def generate_trajectories(self):

        for initial in range(len(self.scenario_set['start'])):
            for goal in range(len(self.scenario_set['end'][initial])):
                trajectories = []
                for trial in range(self.num_trial):
                    # Generating a trajectory for each scenario each trial
                    print("********************************************")
                    print("Trajectory prediction for initial {0}, goal {1}, trial {2}: ".format(initial+1, goal+1, trial+1))
                    traj = self.traj_plan(initial, goal)
                    trajectories.append(traj)
                self.scenario_set['trajs'][initial].append(trajectories)

    def plot(self, background_filename):

        fig, ax = plt.subplots()
        datafile = cbook.get_sample_data(background_filename, asfileobj=False)
        im = image.imread(datafile)
        ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)

        for initial in range(len(self.scenario_set['trajs'])):
            for goal in range(len(self.scenario_set['end'][initial])):
                color = np.random.rand(3,)
                for traj in self.scenario_set['trajs'][initial][goal]:
                    print(self.scenario_set['trajs'][initial])
                    print(self.scenario_set['trajs'][initial][goal])
                    plt.plot(traj[2], traj[1], c=color, alpha=0.6)

        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()
