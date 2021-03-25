#!/usr/bin/env python


import rospy
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
from robot_mover import CommandYoubot


if __name__ == '__main__':
    rospy.init_node('patient_motion_generator', anonymous=True)
    rospy.loginfo("OK")
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

    youbot = CommandYoubot()


    try:

        pub = rospy.Publisher('patient_pose', patient_state, queue_size=1)
        r = rospy.Rate(20)

        pose = Pose()
        state = patient_state()
        robot_state = PointStamped()
        object_state = PointStamped()

        r.sleep()

        patient = sim_patient(patient_motion, patient_traj, env, fall_risk)
        robot = sim_robot(env, robot_state, object_state)
        problem = intervention_problem(fall_risk, env, robot, patient)
        solver_cem = CEMSolver()
        solver_deterministic = DeterministicSolver()
        num_trial = 10
        fallScores = []

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
        # robot_state.point.x = 3.9
        # robot_state.point.y = 5.6
        # robot_state.point.z = 0
        # object_state.point.x = 3.8
        # object_state.point.y = 5.4
        # object_state.point.z = 0

        youbot.set_base_target([robot_state.point.x, robot_state.point.y])

        g_p = []
        pred_trajs = []
        robot_states = []
        object_states = []
        observed_traj = []
        robot_plans = []
        scores = []
        intervention_point = [4, 4, 4]
        assisted = False

        intervention_samples = []
        solutions = []
        assistance = []

        # Uncomment the following to get new plans to simulate

        # patient.sample_traj()
        # for i in range(0,len(patient.sampled_traj)-1):
        #     if i%20==1:
        #         observed_traj.append(patient.sampled_traj[i])
        #         goal_probabilities, predicted_trajs = patient.predict_motion(observed_traj)
        #         g_p.append(goal_probabilities)
        #         most_prob = goal_probabilities.index(max(goal_probabilities))
        #         most_prob_traj = predicted_trajs[most_prob*5]
        #         pred_trajs.append([predicted_trajs[j*5] for j in range(len(goal_probabilities))])
        #         if not assisted:
        #             problem.update(predicted_trajs, robot)
        #             if opt_mode=="exp":
        #                 solution, cost, _, _, _ = solver_cem.optimize(problem.cost_exp, intervention_point)
        #             if opt_mode=="CVaR":
        #                 solution, cost, _, _, _ = solver_cem.optimize(problem.cost_CVaR, intervention_point)
        #             if opt_mode=="exp+CVaR":
        #                 solution, cost, _, _, _ = solver_cem.optimize(problem.cost_exp_CVaR, intervention_point)
        #             if opt_mode=="deterministic":
        #                 solution, cost = solver_deterministic.optimize(problem.cost_deterministic, most_prob_traj)
        #                 print("solution, cost:", solution, cost)
        #
        #             if cost < 10000:
        #                 _ , plan = robot.manipulation_plan(solution)
        #                 print("plan:",plan)
        #                 if plan !=[]:
        #                     robot.execute(plan)
        #                 robot_plans.append(deepcopy(plan))
        #             else:
        #                 solution = deepcopy(robot.robot_state)
        #                 solutions.append([solution.point.x, solution.point.y])
        #                 robot_plans.append([])
        #             robot_states.append(deepcopy(robot.robot_state))
        #             object_states.append(deepcopy(robot.object_state))
        #             if (object_state.point.x-patient.sampled_traj[i][0][0])**2+(object_state.point.y-patient.sampled_traj[i][0][1])**2 <0.3:
        #                 assisted = True
        #
        #         else:
        #             robot_plans.append([])
        #             solution = deepcopy(robot.robot_state)
        #             solutions.append([solution.point.x, solution.point.y])
        #             intervention_samples.append([])
        #             robot_states.append(deepcopy(robot.robot_state))
        #             robot.update_object(patient.sampled_traj[i][0], assisted)
        #             object_states.append(deepcopy(robot.object_state))
        #             scores.append(patient.fall_risk.getDistibutionForTrajectory([patient.sampled_traj[i][0]], False, True)[1][0])
        #
        #     assistance.append(assisted)
        #
        # pickle_res = open("{0}/data/{1}/plan.pickle".format(path, design_name),"wb")
        # pickle.dump([assistance, robot_plans, patient.sampled_traj], pickle_res)
        # pickle_res.close()
        # plot_prediction(background_filename, observed_traj, g_p, pred_trajs, intervention_samples, solutions, robot_states, robot_plans, object_states, patient.motion.ends[patient.initial], patient.env, 0)
        pickle_res = open("{0}/data/{1}/plan.pickle".format(path, design_name),"rb")
        assistance, robot_plans, traj = pickle.load(pickle_res)
        pickle_res.close()
        patient.sampled_traj = traj

        j = 0

        for i in range(0,len(patient.sampled_traj)-1):
            if i%20==1:
                if robot_plans[j] !=[]:
                    youbot.set_base_target(robot_plans[j][1])
                j+=1

            phi = patient.sampled_traj[i][0][3]
            if phi<=-np.pi:
                phi += 2*np.pi
            elif phi>=np.pi:
                phi -= 2*np.pi

            pose.position.x = patient.sampled_traj[i][0][1]
            pose.position.y = patient.sampled_traj[i][0][0]
            pose.position.z = patient.sampled_traj[i][0][2]
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = phi
            pose.orientation.w = 0

            state.pose = pose
            state.activity = patient.sampled_traj[i][1]
            pub.publish(state)

            youbot.calculate_vel(assistance[i], patient.sampled_traj[i][0])

            r.sleep()

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
