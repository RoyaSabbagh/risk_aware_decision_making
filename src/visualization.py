from scipy import *
from scipy.linalg import norm, pinv
import numpy as np
from matplotlib import pyplot as plt
import math
import random
from shapely.geometry import Polygon, Point
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.cbook as cbook
import matplotlib.image as image
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plot
from matplotlib import animation
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import numpy as np
from sklearn.metrics import r2_score
from utility import find_corners
from scipy.stats import norm, gamma, rayleigh, exponweib
import matplotlib.mlab as mlab


def plot_metrics():

    plt.rcParams.update({'font.size':26})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    #setting up the plot ax and fig
    colors = [[200/float(255),160/float(255),40/float(255)], [112/float(255), 204/float(255), 225/float(255)], [200/float(255), 80/float(255), 40/float(255)], [130/float(255), 200/float(255), 80/float(255)], [54/float(255),50/float(255), 140/float(255)]]
    fig,ax = plt.subplots(1, figsize=(15,6))
    width = 0.1

    exp_means = [4.2301860231272, 5.913256421983376, 7.3697397748221265, 4.307270114512634]
    CVaR_means = [3.739367583126901, 5.63510035092322, 6.421855267149385, 4.202965453676939]
    expCVaR_means = [3.733835578440041, 6.035561420558764, 6.473164174883367, 4.367714098414838]
    det_means = [4.474670401267435, 5.80326711118155, 7.3901195084453, 4.511138074223044]
    no_means = [4.6008153562182073, 6.3241073697544286, 7.3815376694730785, 4.377146522422491]
    exp_CVaR = [7.168313416096223, 10.522884868265773, 10.941960784313723, 7.890275833944571]
    CVaR_CVaR = [5.7352802576332005, 9.978177958570113, 9.987736185383241, 7.33672073853746]
    expCVaR_CVaR = [5.8312338641750445, 10.411983795170956, 10.280522875816992, 7.877219385235939]
    det_CVaR = [7.470788579612111, 10.470251565428033, 11.651597830621606, 8.048820280463262]
    no_CVaR = [7.579933791698498, 11.066684155651966, 10.881381074168797, 8.012252595652629]


    x = 4*np.arange(4)  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - 2*width, expCVaR_means, width, label=r'$\mathrm{Expected+CVaR~Cost}$', color=colors[0])
    rects2 = ax.bar(x - 1*width, exp_means, width, label=r'$\mathrm{CVaR~Cost}$', color=colors[1])
    rects3 = ax.bar(x + 0*width, CVaR_means, width, label=r'$\mathrm{Expected~Cost}$', color=colors[2])
    rects4 = ax.bar(x + 1*width, det_means, width, label=r'$\mathrm{Deterministic}$', color=colors[3])
    rects5 = ax.bar(x + 2*width, no_means, width, label=r'$\mathrm{No~Intervention}$', color=colors[4])

    ax.set_xticklabels([r'$\mathrm{Bed~(Right)}$', r'$\mathrm{Toilet}$', r'$\mathrm{Bed~(Left)}$', r'$\mathrm{Visitor~Chair}$'], fontsize=20)

    ax.set_xticks([0, 4, 8 , 12])
    # ax.set_xlabel(r'$\mathrm{Fall~Score}$', fontsize=20)
    ax.set_ylabel(r'$\mathrm{Fall~Scores~Mean}$', fontsize=20)
    ax.legend(fontsize=16);
    ax.grid(True)
    ax.set_ylim(3, 8)

    fig2,ax2 = plt.subplots(1, figsize=(15,6))
    rects6 = ax2.bar(x - 2*width, expCVaR_CVaR, width, label=r'$\mathrm{Expected+CVaR~Cost}$', color=colors[0])
    rects7 = ax2.bar(x - 1*width, CVaR_CVaR, width, label=r'$\mathrm{CVaR~Cost}$', color=colors[1])
    rects8 = ax2.bar(x + 0*width, exp_CVaR, width, label=r'$\mathrm{Expected~Cost}$', color=colors[2])
    rects9 = ax2.bar(x + 1*width, det_CVaR, width, label=r'$\mathrm{Deterministic}$', color=colors[3])
    rects10 = ax2.bar(x + 2*width, no_CVaR, width, label=r'$\mathrm{No~Intervention}$', color=colors[4])


    ax2.set_xticklabels([r'$\mathrm{Bed~(Right)}$', r'$\mathrm{Toilet}$', r'$\mathrm{Bed~(Left)}$', r'$\mathrm{Visitor~Chair}$'], fontsize=20)
    ax2.set_xticks([0, 4, 8 , 12])
    # ax.set_xlabel(r'$\mathrm{Fall~Score}$', fontsize=20)
    ax2.set_ylabel(r'$\mathrm{Fall~Scores~CVaR}$', fontsize=20)
    ax2.legend(fontsize=16);
    ax2.grid(True)
    ax2.set_ylim(5, 12)

    plt.show()



def plot_fall_distributions(fallScores):

    plt.rcParams.update({'font.size':26})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    fig, axs = plt.subplots(5, 1)
    fig.set_size_inches(10, 20)
    lines = []
    bins = [[],[],[],[],[]]
    y = [[],[],[],[],[]]
    y2 = [[],[],[],[],[]]
    fallScores[4] = [fallScores[4][i]/0.8 for i in range(len(fallScores[4]))]

    labels = [r'$\mathrm{Deterministic}$', r'$\mathrm{Probabilistic-CVaR~Cost}$', r'$\mathrm{Probabilistic-Expected+CVaR~Cost}$', r'$\mathrm{Probabilistic-Expected~Cost}$', r'$\mathrm{No~Intervention}$']

    for i in range(5):
        # (mu, sigma) = rayleigh.fit(fallScores[i])
        prameters = exponweib.fit(fallScores[i], floc=0)
        # print(mu,sigma)
        n, bins[i], patches = axs[i].hist(fallScores[i],  density=True, stacked=True, color = "royalblue", bins = 40, alpha=1)
        # y[i] = rayleigh.pdf(bins[i], mu, sigma)
        y2[i] = exponweib.pdf(bins[i], *prameters)
        # axs[i].plot(bins[i], y[i], "red", linewidth=3)
        axs[i].plot(bins[i], y2[i], "black", linewidth=3)

        axs[i].grid(True)
        axs[i].set_xlim(0, 17)
        axs[i].set_ylim(0, 1)
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([r'$\mathrm{0.0}$', r'$\mathrm{0.5}$', r'$\mathrm{1.0}$'], fontsize=20)
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        axs[i].text(0.7, 0.9, labels[i], transform=axs[i].transAxes, fontsize=14,
        verticalalignment='top', bbox=props, ha='center')


    axs[4].set_xticklabels([r'$\mathrm{0}$', r'$\mathrm{2.5}$', r'$\mathrm{5.0}$', r'$\mathrm{7.5}$', r'$\mathrm{10.0}$', r'$\mathrm{12.5}$', r'$\mathrm{15}$'], fontsize=20)
    axs[4].set_xlabel(r'$\mathrm{Fall~Score}$', fontsize=20)
    axs[2].set_ylabel(r'$\mathrm{Density}$', fontsize=20)


    plt.show()

    fig5, ax6 = plt.subplots()
    fig5.set_size_inches(15, 10)
    labels = [r'$\mathrm{Deterministic}$', r'$\mathrm{Probabilistic-CVaR~Cost}$', r'$\mathrm{Probabilistic-Expected+CVaR~Cost}$', r'$\mathrm{Probabilistic-Expected~Cost}$', r'$\mathrm{No~Intervention}$']
    colors = ['k-.', 'b--', 'r', 'b:', "k"]

    for i in range(5):
        lines.append(ax6.plot(bins[i], y2[i], colors[i], linewidth=3, label=labels[i]))

    # Add labels
    ax6.set_xticklabels([r'$\mathrm{}$', r'$\mathrm{2}$', r'$\mathrm{4}$', r'$\mathrm{6}$', r'$\mathrm{8}$', r'$\mathrm{10}$', r'$\mathrm{12}$', r'$\mathrm{14}$',r'$\mathrm{16}$'], fontsize=20)
    ax6.set_yticklabels([r'$\mathrm{0.00}$', r'$\mathrm{0.05}$', r'$\mathrm{0.10}$', r'$\mathrm{0.15}$', r'$\mathrm{0.20}$', r'$\mathrm{0.25}$', r'$\mathrm{0.30}$'], fontsize=20)

    ax6.set_xlabel(r'$\mathrm{Fall~Score}$', fontsize=20)
    ax6.set_ylabel(r'$\mathrm{Density}$', fontsize=20)
    ax6.legend(fontsize=20);
    ax6.grid(True)

    plt.show()


def plot_prediction(background_filename, traj, goal_probabilities, predicted_trajs, intervention_samples, solutions, robot_states, robot_plans, object_states, goals, env, trial):
    matplotlib.use("Agg")
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(1.5, 7)
    ax.set_ylim(3, 7)
    # ax.set_title('Intention Prediction')

    dt = 0.5
    n = len(traj)
    time_pred = np.linspace(0, dt*n, n)

    datafile = cbook.get_sample_data(background_filename, asfileobj=False)
    im = image.imread(datafile)
    ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)

    x = [traj[i][0][1] for i in range(len(traj))]
    y = [traj[i][0][0] for i in range(len(traj))]

    x_r = [robot_states[i].point.y for i in range(len(robot_states))]
    y_r = [robot_states[i].point.x for i in range(len(robot_states))]


    x_plan = []
    y_plan = []
    for j in range(len(robot_plans)):
        x_plan.append([robot_plans[j][i][1] for i in range(len(robot_plans[j]))])
        y_plan.append([robot_plans[j][i][0] for i in range(len(robot_plans[j]))])

    data = []
    n_samples = []
    for j in range(len(intervention_samples)):
        if not n_samples:
            n_samples.append(len(intervention_samples[j]))
        else:
            n_samples.append(len(intervention_samples[j])+n_samples[j-1])
        samples = []
        for k in range(len(intervention_samples[j])):
            points = [[intervention_samples[j][k][i][1], intervention_samples[j][k][i][0]] for i in range(len(intervention_samples[j][k]))]
            samples.append(points)
        data.append(samples)

    goal_markers = []
    predicted_traj_lines = []

    for goal in goals:
        center = env.sample_zones[goal].centroid
        goal_markers.append(ax.scatter(center.y,center.x, s= 120, marker="X", color='darkmagenta', alpha = 0.1))
        predicted_traj_line, = ax.plot([], [], 'b--', lw = 3, alpha  = 0.1)
        predicted_traj_lines.append(predicted_traj_line)


    # sampled_interventions = ax.scatter([],[], s= 50, marker=".", color='blue', alpha = 0.5)

    # robot_marker = ax.scatter([],[], s= 150, marker="o", color='green', alpha = 1)

    patient_marker = ax.scatter([],[], s= 100, marker="D", color='k', alpha = 1)

    # solution_marker = ax.scatter([],[], s= 100, marker="o", facecolors='none', edgecolors='r', alpha = 1 , linewidth=2)

    # corners = find_corners(object_states[0].point.y, object_states[0].point.x, object_states[0].point.z, 0.2, 0.3)
    # Walker = plt.Polygon(corners, closed=None, fill=None, edgecolor='k', alpha = 1, linewidth=2)
    # ax.add_artist(Walker)


    traj_line, = ax.plot([], [], lw = 3, color='b')
    robot_traj_line, = ax.plot([], [], lw = 3, color='g')
    plan_line, = ax.plot([], [], 'g--', lw = 3, alpha  = 0.5)

    def animate(k):
        # lb = 0
        # for k in range(len(n_samples)):
        #     if i>=lb and i<n_samples[k]:
        #         break
        #     else:
        #         lb = n_samples[k]
        traj_line.set_data(x[0:k+1], y[0:k+1])
        # robot_traj_line.set_data(x_r[0:k+1], y_r[0:k+1])
        # plan_line.set_data(x_plan[k], y_plan[k])

        # sampled_interventions.set_offsets(data[k][i-lb])
        # robot_marker.set_offsets([robot_states[k].point.y, robot_states[k].point.x])
        patient_marker.set_offsets([x[k], y[k]])
        # solution_marker.set_offsets([solutions[k][1], solutions[k][0]])

        # corners = find_corners(object_states[k].point.y, object_states[k].point.x, object_states[k].point.z, 0.2, 0.3)
        # Walker.set_xy(corners)

        for j in range(len(goals)):
            goal_markers[j].set_alpha(0.1+0.9*goal_probabilities[k][j])
            x_p = [predicted_trajs[k][j][l][1] for l in range(len(predicted_trajs[k][j])-1)]
            y_p = [predicted_trajs[k][j][l][0] for l in range(len(predicted_trajs[k][j])-1)]
            predicted_traj_lines[j].set_data(x_p, y_p)
            if goal_probabilities[k][j]<0.1:
                # al = 0
                al = 0.1+0.5*goal_probabilities[k][j]
            else:
                al = 0.1+0.5*goal_probabilities[k][j]
            predicted_traj_lines[j].set_alpha(al)
        return predicted_traj_lines + [traj_line]

    anim = FuncAnimation(fig, animate, frames=len(solutions), interval=100)
    plt.show()
    anim.save('results/test-{}.mp4'.format(trial), writer = 'ffmpeg', fps = 1)


def plot_intent_prediction(background_filename, traj, goal_probabilities, goals, env, trial):
    matplotlib.use("Agg")
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(1.5, 7)
    ax.set_ylim(3, 7)
    ax.set_title('Intention Prediction')

    dt = 0.5
    n = len(traj)
    time_pred = np.linspace(0, dt*n, n)

    datafile = cbook.get_sample_data(background_filename, asfileobj=False)
    im = image.imread(datafile)
    ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)

    x = [traj[i][1] for i in range(n)]
    y = [traj[i][0] for i in range(n)]
    print(goals)
    goal_markers = []
    for goal in goals:
        center = env.sample_zones[goal].centroid
        goal_markers.append(ax.scatter(center.y,center.x, s= 120, marker="X", color='darkmagenta', alpha = 0.1))

    traj_line, = ax.plot([], [], lw = 3, color='darkcyan')
    def animate(i):
        traj_line.set_data(x[0:i], y[0:i])
        for j in range(len(goals)):
            goal_markers[j].set_alpha(0.1+0.9*goal_probabilities[j][i])
        return traj_line

    anim = FuncAnimation(fig, animate, frames=len(traj)-1, interval=100)
    # plt.show()
    anim.save('results/test-{}.mp4'.format(trial), writer = 'ffmpeg', fps = 10)

def plot_paths(background_filename, predicted_paths, color):

    dt = 0.5
    n = len(predicted_paths[0][0])
    time_pred = np.linspace(0, dt*n, n)

    fig, ax = plt.subplots()
    datafile = cbook.get_sample_data(background_filename, asfileobj=False)
    im = image.imread(datafile)
    ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)
    for traj in predicted_paths:
        x_pred = [traj[0][i][1] for i in range(n)]
        y_pred = [traj[0][i][0] for i in range(n)]
        plt.plot(x_pred, y_pred, c = color, alpha=0.3)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.figure(2)
    for traj in predicted_paths:
        x_pred = [traj[0][i][0] for i in range(n)]
        plt.plot(time_pred, x_pred, color='b')
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')

    plt.figure(3)
    for traj in predicted_paths:
        y_pred = [traj[0][i][1] for i in range(n)]
        plt.plot(time_pred, y_pred, color='b')
    plt.xlabel('time (s)')
    plt.ylabel('y (m)')
    plt.show()

def plot_pred(background_filename, predicted_path, generated_trajectories=[]):

    dt = 0.5
    n = len(predicted_path)
    time_pred = np.linspace(0, dt*n, n)
    x_pred = []
    y_pred = []
    std_x_pred = []
    std_y_pred = []
    for point in predicted_path:
        x_pred.append(point[0][0])
        y_pred.append(point[0][1])
        std_x_pred.append(point[1][0][0])
        std_y_pred.append(point[1][1][1])


    fig, ax = plt.subplots()
    datafile = cbook.get_sample_data(background_filename, asfileobj=False)
    im = image.imread(datafile)
    ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)
    plt.plot(y_pred, x_pred, '--r', linewidth = 2)
    for traj in generated_trajectories:
        plt.plot(traj[2], traj[1], 'gray', alpha=0.3)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.figure(2)
    for traj in generated_trajectories:
        time_gen = np.linspace(0, dt*len(traj[2]), len(traj[2]))
        plt.plot(time_gen, traj[2], '.', color='b')
    plt.plot(time_pred, y_pred, '-', color='r')
    plt.fill_between(time_pred, [y_pred[i] - 2 * std_y_pred[i] for i in range(n)], [y_pred[i] + 2 * std_y_pred[i] for i in range(n)], color='gray', alpha=0.3)
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')

    plt.figure(3)
    for traj in generated_trajectories:
        time_gen = np.linspace(0, dt*len(traj[2]), len(traj[2]))
        plt.plot(time_gen, traj[1], '.', color='b')
    plt.plot(time_pred, x_pred, '-', color='r')
    plt.fill_between(time_pred, [x_pred[i] - 2 * std_x_pred[i] for i in range(n)], [x_pred[i] + 2 * std_x_pred[i] for i in range(n)], color='gray', alpha=0.3)
    plt.xlabel('time (s)')
    plt.ylabel('y (m)')
    plt.show()
