from gurobipy import *
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.linalg import norm, pinv


def OptPath(state_s, state_f, obstacles, T):
# Create a new model
    big_M = 1000000000000000
    dt = 0.5
    n = int(float(T)/dt)+1
    v_max = [0.4, 3]
    command = []
    phi = []
    path = []
    cost = float('inf')
    initial_orientation = 0

    PathPlanner = Model("Path")
# Create variables
    PathPlanner.setParam('OutputFlag', 0)
    PathPlanner.setParam("TimeLimit", 10)
    states = PathPlanner.addVars(n+1, 2, lb=-10, ub=10, vtype=GRB.CONTINUOUS, name="states")
    dstates = PathPlanner.addVars(n, 2, lb=-10, ub=10, vtype=GRB.CONTINUOUS, name="dstates")
    v_abs = PathPlanner.addVars(n, lb=0, ub=10, vtype=GRB.CONTINUOUS, name="v_abs")
    u =PathPlanner.addVars(len(obstacles), n+1, 4, lb=-1, ub=1, vtype=GRB.BINARY, name="u")
# Set objective
    PathPlanner.setObjective(sum([(states[k,l]-state_f[l])*(states[k,l]-state_f[l]) for k in range(n) for l in range(2)]), GRB.MINIMIZE)
# Add position, velocity and acceleration constraints
    for i in range(2):
        PathPlanner.addConstr(states[0, i] == state_s[i], "c0_{}".format(i))
        PathPlanner.addConstr(states[n, i] == state_f[i], "c00_{}".format(i))
    for j in range(n):
        for i in range(2):
            PathPlanner.addConstr(dstates[j,i] == (states[j+1, i]-states[j, i]), "c1_{}_{}".format(i,j))
        PathPlanner.addConstr(v_max[0] * dt * v_max[0] * dt >= (dstates[j,1] * dstates[j,1]) + (dstates[j,0] * dstates[j,0]), "c2_{}".format(j))
# Add Obstacle constraints
    for n_obs in range(len(obstacles)):
        for i in range(0, n+1):
            for k in range(4):
                PathPlanner.addConstr((states[i, 1] - obstacles[n_obs][0][k]*states[i, 0] - obstacles[n_obs][1][k] )*(obstacles[n_obs][2][1] - obstacles[n_obs][0][k]*obstacles[n_obs][2][0] - obstacles[n_obs][1][k] ) <= - (u[n_obs, i, k]-1)*big_M, "c3_{}_{}_{}".format(n_obs,i,k))
        for i in range(0,n):
            PathPlanner.addConstr(sum(u[n_obs, i, ii] for ii in range(0, 4)) >= 1, "c4_{}_{}".format(n_obs, i))

    PathPlanner.optimize()
    if PathPlanner.Status == 2:
        cost = PathPlanner.objVal
        phi = [state_s[2]]
        for i in range(n):
            x1 = states[i,0].x
            y1 = states[i,1].x
            x2 = states[i+1,0].x
            y2 = states[i+1,1].x
            new_phi = math.atan2(y2-y1,x2-x1)
            phi.append(new_phi)
            path.append([states[i,0].x, states[i,1].x, phi[-1]])

    return cost, path

def OptPath_patient(state_s, state_f, v_max, obstacles, n, assistive_device):
    dt = 0.5
    big_M = 100000000
    if assistive_device:
        w = [0, 1, 0]
    else:
        w = [0, 1, 0]
    # Create a new model
    Patient_PathPlanner = Model("Patient_path")
    Patient_PathPlanner.reset(0)
    Patient_PathPlanner.setParam("TimeLimit", 30.0)
    Patient_PathPlanner.setParam('OutputFlag', 0)
    # Create variables
    states = Patient_PathPlanner.addVars(n, 2, lb=0, ub=10, vtype=GRB.CONTINUOUS, name="states")
    dstates = Patient_PathPlanner.addVars(n-1, 2, lb=-5, ub=5, vtype=GRB.CONTINUOUS, name="dstates")
    dist_min = Patient_PathPlanner.addVars(n, lb=0, ub=20, vtype=GRB.CONTINUOUS, name="dist_min")
    dist = Patient_PathPlanner.addVars(n-1, len(obstacles), lb=0, ub=20, vtype=GRB.CONTINUOUS, name="dist")
    dist_x = Patient_PathPlanner.addVars(n-1, len(obstacles), lb=-10, ub=10, vtype=GRB.CONTINUOUS, name="distx")
    dist_x_abs = Patient_PathPlanner.addVars(n-1, len(obstacles), lb=0, ub=10, vtype=GRB.CONTINUOUS, name="distxabs")
    dist_y = Patient_PathPlanner.addVars(n-1, len(obstacles), lb=-10, ub=10, vtype=GRB.CONTINUOUS, name="disty")
    dist_y_abs = Patient_PathPlanner.addVars(n-1, len(obstacles), lb=0, ub=10, vtype=GRB.CONTINUOUS, name="distyabs")
    u = Patient_PathPlanner.addVars(len(obstacles), n+1, 4, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.BINARY, name="u")
    # Set objective
    Patient_PathPlanner.setObjective( w[0]*sum([dist_min[i]*dist_min[i] for i in range(n)]) +w[1]*sum([(states[i,k]-state_f[k])*(states[i,k]-state_f[k]) for k in range(2) for i in range(n-1)]), GRB.MINIMIZE)# +w[2]* sum([weights[j,i] * weights[j,i] for i in range(2) for j in range(x_rbf.n_rbf)]) , GRB.MINIMIZE)

    # Add position, velocity and acceleration constraints
    for i in range(2):
        Patient_PathPlanner.addConstr(states[0, i] == state_s[i])
        Patient_PathPlanner.addConstr(states[n-1, i] == state_f[i])
        Patient_PathPlanner.addConstr(dstates[0, i] == 0)
        # for j in range(n):
        for j in range(n-1):
            Patient_PathPlanner.addConstr(dstates[j,i] == (states[j+1, i]-states[j, i]))
    for j in range(n-1):
        Patient_PathPlanner.addConstr(v_max[0] * dt * v_max[0] * dt>= (dstates[j,1] * dstates[j,1]) + (dstates[j,0] * dstates[j,0]))
    # Add Obstacle constraints
    for n_obs in range(len(obstacles)):
        for i in range(0, n):
            for k in range(4):
                Patient_PathPlanner.addConstr((states[i, 1] - obstacles[n_obs][0][k]*states[i, 0] - obstacles[n_obs][1][k] )*(obstacles[n_obs][2][1] - obstacles[n_obs][0][k]*obstacles[n_obs][2][0] - obstacles[n_obs][1][k]) <= - (u[n_obs, i, k]-1)*big_M)
        for i in range(0,n):
            Patient_PathPlanner.addConstr(sum(u[n_obs, i, ii] for ii in range(0, 4)) >= 1)
    # Define minimum distance to an external supporting point
    for i in range(1,n-1):
        for n_obs in range(len(obstacles)):
            Patient_PathPlanner.addConstr(dist_x[i,n_obs]==states[i, 0]- obstacles[n_obs][2][0])
            Patient_PathPlanner.addGenConstrAbs(dist_x_abs[i,n_obs], dist_x[i,n_obs])
            Patient_PathPlanner.addConstr(dist_y[i,n_obs]==states[i,1]- obstacles[n_obs][2][1])
            Patient_PathPlanner.addGenConstrAbs(dist_y_abs[i,n_obs], dist_y[i,n_obs])
            Patient_PathPlanner.addConstr(dist[i,n_obs] == dist_x_abs[i,n_obs]+dist_y_abs[i,n_obs])
        Patient_PathPlanner.addConstr(dist_min[i] == min_([dist[i,n_obs] for n_obs in range(len(obstacles))]))


    Patient_PathPlanner.optimize()
    path = []
    cost = float('inf')
    if Patient_PathPlanner.Status == 2:
        cost = Patient_PathPlanner.objVal
        phi = [state_s[2]]
        for i in range(n-1):
            x1 = states[i,0].x
            y1 = states[i,1].x
            x2 = states[i+1,0].x
            y2 = states[i+1,1].x
            new_phi = math.atan2(x2-x1, y2-y1)
            if new_phi > np.pi:
                new_phi-=2*np.pi
            elif new_phi < -np.pi:
                new_phi+=2*np.pi
            v = math.sqrt((x2-x1)**2+(y2-y1)**2)/dt
            w = (new_phi-phi[-1])/dt
            phi.append(new_phi)
            if (states[i,0].x-states[i+1,0].x)**2>0.000001 or (states[i,1].x-states[i+1,1].x)**2>0.000001:
                path.append([states[i,0].x, states[i,1].x, 0.6, phi[-1]+np.pi/2, v, w])

        path.append([states[i,0].x, states[i,1].x, 0.6, phi[-1]+np.pi/2, v, w])

    return dt*n, cost, path, Patient_PathPlanner.Status
