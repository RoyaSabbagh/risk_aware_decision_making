#!/usr/bin/env python

from Fall_risk_assesment import Environment_Image, FallRiskAssesment
import numpy as np
from scipy import *
from Gaussian_processes import MGPR
import random
import pickle
from GPs_learning import motion_set
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.utilities import to_default_float
from visualization import plot_prediction, plot_fall_distributions
from simulation import sim_patient, sim_robot
from geometry_msgs.msg import Pose, PointStamped
from scipy.spatial.transform import Rotation
from risk_aware_planning.msg import patient_state
from intervention_problem import *
from solvers import *
from copy import deepcopy


if __name__ == '__main__':

    design_name = "Room-2-Inboard-Footwall"
    path = "/home/roya/catkin_ws/src/risk_aware_planning/src"
    background_filename = "{0}/Room_Designs/{1}_objects_rotated.png".format(path, design_name)

    pickle_env = open("{0}/data/{1}/env.pickle".format(path,design_name),"rb")
    env = pickle.load(pickle_env)
    pickle_env.close()

    pickle_baseline_evaluation = open("{0}/data/{1}/baseline_evaluation.pickle".format(path,design_name),"rb")
    fall_risk = pickle.load(pickle_baseline_evaluation)
    pickle_baseline_evaluation.close()

    pickle_trajectories = open("{0}/data/{1}/trajectories.pickle".format(path,design_name),"rb")
    patient_traj = pickle.load(pickle_trajectories)
    pickle_trajectories.close()

    pickle_motion = open("{0}/data/{1}/motion_GPs.pickle".format(path,design_name),"rb")
    patient_motion = pickle.load(pickle_motion)
    pickle_motion.close()

    robot_state = PointStamped()
    object_state = PointStamped()

    patient = sim_patient(patient_motion, patient_traj, env, fall_risk)
    robot = sim_robot(env, robot_state, object_state)
    problem = intervention_problem(fall_risk, env, robot, patient)
    solver_cem = CEMSolver()
    solver_deterministic = DeterministicSolver()
    num_trial = 10
    fallScores = []

    experiment_mode = 0

    if experiment_mode == 0:
        # Experiments to get risk-aware planning video
        # opt_mode = "exp"
        # opt_mode = "CVaR"
        # opt_mode = "exp+CVaR"
        opt_mode = "deterministic"

        robot_state.point.x = 6.1
        robot_state.point.y = 2.9
        robot_state.point.z = 0
        object_state.point.x = 6
        object_state.point.y = 2.8
        object_state.point.z = 0
        #
        # robot_state.point.x = 3.7
        # robot_state.point.y = 5.4
        # robot_state.point.z = 0
        # object_state.point.x = 3.6
        # object_state.point.y = 5.2
        # object_state.point.z = 0
        patient.sample_traj()
        g_p = []
        pred_trajs = []
        robot_states = []
        object_states = []
        observed_traj = []
        robot_plans = []
        intervention_samples = []
        solutions = []
        scores = []
        intervention_point = [4, 4, 4]
        assisted = False

        for i in range(0,len(patient.sampled_traj)-1):
            if i%10==1:
                if opt_mode=="no-intervetion":
                    scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, False)[1][0])
                else:
                    observed_traj.append(patient.sampled_traj[i])
                    goal_probabilities, predicted_trajs = patient.predict_motion(observed_traj)
                    g_p.append(goal_probabilities)
                    most_prob = goal_probabilities.index(max(goal_probabilities))
                    most_prob_traj = predicted_trajs[most_prob*5]
                    pred_trajs.append([predicted_trajs[j*5] for j in range(len(goal_probabilities))])
                    if not assisted:
                        problem.update(predicted_trajs, robot)
                        if opt_mode=="exp":
                            solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_exp, intervention_point)
                        if opt_mode=="CVaR":
                            solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_CVaR, intervention_point)
                        if opt_mode=="exp+CVaR":
                            solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_exp_CVaR, intervention_point)
                        if opt_mode=="deterministic":
                            solution, cost = solver_deterministic.optimize(problem.cost_deterministic, most_prob_traj)
                            print("solution, cost:", solution, cost)
                            sample_iterates = [[solution]]
                        intervention_samples.append([sample_iterates[-1]])

                        if cost < 10000:
                            solutions.append(solution)
                            _ , plan = robot.manipulation_plan(solution)
                            print("plan:",plan)
                            if plan !=[]:
                                robot.execute(plan)
                            robot_plans.append(deepcopy(plan))
                        else:
                            solution = deepcopy(robot.robot_state)
                            solutions.append([solution.point.x, solution.point.y])
                            robot_plans.append([])
                        robot_states.append(deepcopy(robot.robot_state))
                        object_states.append(deepcopy(robot.object_state))
                        if (object_state.point.x-patient.sampled_traj[i][0][0])**2+(object_state.point.y-patient.sampled_traj[i][0][1])**2 <0.4:
                            assisted = True


                        scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, False)[1][0])
                    else:
                        robot_plans.append([])
                        solution = deepcopy(robot.robot_state)
                        solutions.append([solution.point.x, solution.point.y])
                        intervention_samples.append([])
                        robot_states.append(deepcopy(robot.robot_state))
                        robot.update_object(patient.sampled_traj[i][0], assisted)
                        object_states.append(deepcopy(robot.object_state))
                        scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, True)[1][0])
        # print(scores)
        # pickle_res = open("{0}/data/{1}/results-{2}.pickle".format(path, design_name, opt_mode),"wb")
        # pickle.dump([background_filename, observed_traj, g_p, pred_trajs, intervention_samples, solutions, robot_states, robot_plans, object_states, patient.motion.ends[patient.initial], patient.env, 0], pickle_res)
        # pickle_res.close()
        plot_prediction(background_filename, observed_traj, g_p, pred_trajs, intervention_samples, solutions, robot_states, robot_plans, object_states, patient.motion.ends[patient.initial], patient.env, 0)

    if experiment_mode == 1:

        # Experiments to get fall risk distribution
        for opt_mode in ["exp", "CVaR", "exp+CVaR", "deterministic", "no-intervention"]:

            scores = []
            for k in range(num_trial):
                print("******************{0} - Trial {1}*******************".format(opt_mode, k))
                # robot_state.point.x = 6.1
                # robot_state.point.y = 2.9
                # robot_state.point.z = 0
                # object_state.point.x = 6
                # object_state.point.y = 2.8
                # object_state.point.z = 0
                robot_state.point.x = 3.9
                robot_state.point.y = 5.4
                robot_state.point.z = 0
                object_state.point.x = 3.8
                object_state.point.y = 5.2
                object_state.point.z = 0
                patient.sample_traj()
                g_p = []
                pred_trajs = []
                robot_states = []
                object_states = []
                observed_traj = []
                intervention_samples = []
                intervention_point = [4, 4, 4]
                assisted = False

                for i in range(0,len(patient.sampled_traj)-1):
                    if i%20==1:
                        if opt_mode=="no-intervetion":
                            scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, False)[1][0])
                        else:
                            observed_traj.append(patient.sampled_traj[i])
                            goal_probabilities, predicted_trajs = patient.predict_motion(observed_traj)
                            g_p.append(goal_probabilities)
                            most_prob = goal_probabilities.index(max(goal_probabilities))
                            pred_trajs.append(predicted_trajs[most_prob*5])
                            if not assisted:
                                problem.update(predicted_trajs, robot)
                                if opt_mode=="exp":
                                    solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_exp, intervention_point)
                                if opt_mode=="CVaR":
                                    solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_CVaR, intervention_point)
                                if opt_mode=="exp+CVaR":
                                    solution, cost, converged, iterates, sample_iterates = solver_cem.optimize(problem.cost_exp_CVaR, intervention_point)
                                if opt_mode=="deterministic":
                                    solution, cost = solver_deterministic.optimize(problem.cost_deterministic, pred_trajs[-1])
                                    sample_iterates = [[solution]]
                                intervention_samples.append(sample_iterates)

                                _ , plan = robot.manipulation_plan(solution)
                                if plan !=[]:
                                    robot.execute(plan)
                                robot_states.append(deepcopy(robot.robot_state))
                                object_states.append(deepcopy(robot.object_state))
                                if (object_state.point.x-patient.sampled_traj[i][0][0])**2+(object_state.point.y-patient.sampled_traj[i][0][1])**2 <0.2:
                                    assisted = True

                                scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, False)[1][0])
                            else:
                                intervention_samples.append([sample_iterates[-1]])
                                robot_states.append(deepcopy(robot.robot_state))
                                robot.update_object(patient.sampled_traj[i][0], assisted)
                                object_states.append(deepcopy(robot.object_state))
                                scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, True)[1][0])

            fallScores.append(scores)

        print(fallScores)
        pickle_fall = open("{0}/data/{1}/fallScores_door.pickle".format(path,design_name),"wb")
        pickle.dump(fallScores, pickle_fall)
        pickle_fall.close()
        plot_fall_distributions(fallScores)
