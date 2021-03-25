#!/usr/bin/env python
# license removed for brevity
import rospy
import copy
import math
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from scipy.spatial.transform import Rotation

class CommandYoubot():

    def __init__(self):
        self.base_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.arm_pub = rospy.Publisher('/arm_1/arm_controller/command', JointTrajectory, queue_size=1)
        self.gripPub = rospy.Publisher('/arm_1/gripper_controller/command', JointTrajectory, queue_size=1)
        self.walkerPub = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.offset = [-(5.33 - 5.6),
                       -(4.03 - 3.8),
                       0    - 0.1,
                       0    - 0  ,
                       0    - 0  ,
                       0.4  - 0]

        self.current_base = []
        self.base_cmd = Twist()
        self.base_target=[5.4,4.2]
        self.eps = 0.01
        self.v_max = 0.15
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.base_callback, queue_size=1)


        self.arm_targets = JointTrajectory()
        arm_target1 = JointTrajectoryPoint()

        self.gripper_targets = JointTrajectory()
        gripper_targets1 = JointTrajectoryPoint()

        self.arm_targets.joint_names = ["arm_joint_1","arm_joint_2","arm_joint_3","arm_joint_4","arm_joint_5"]
        self.gripper_targets.joint_names = ["gripper_finger_joint_l","gripper_finger_joint_r"]

        arm_target1.positions = [0.03,0.5,-2.0,0.9,0.0]
        arm_target1.velocities = [0.5,0.5,0.5,0.5,0.5]
        arm_target1.accelerations = [0.2,0.2,0.2,0.2,0.2]
        arm_target1.effort = [100]
        arm_target1.time_from_start = rospy.Duration(0.01)

        gripper_targets1.positions = [0.01, 0.01]
        gripper_targets1.velocities = [0, 0]
        gripper_targets1.accelerations = [0, 0]
        gripper_targets1.effort = []
        gripper_targets1.time_from_start = rospy.Duration(0.01)

        self.arm_targets.points.append(arm_target1)
        self.gripper_targets.points.append(gripper_targets1)

    def set_base_target(self, target):
        self.base_target = target

    def calculate_vel(self, assisted, patient_pose):
        inc = [self.base_target[1] - self.current_base[0], self.base_target[0] - self.current_base[1]]
        d = np.sqrt(inc[0]**2+ inc[1]**2)
        phi = np.arctan2(float(inc[1]), inc[0])
        if d > self.eps:
            lin_vel = self.v_max
        else: lin_vel = 0
        # print("self.base_target:", self.base_target)
        # print("self.current_base:", self.current_base)
        # print("vel:", lin_vel)


        self.arm_pub.publish(self.arm_targets)
        self.gripPub.publish(self.gripper_targets)

        if not assisted:

            self.base_cmd.linear.x = lin_vel*np.cos(phi)
            self.base_cmd.linear.y = lin_vel*np.sin(phi)
            self.base_cmd.linear.z = 0.0
            self.base_cmd.angular.x = 0.0
            self.base_cmd.angular.y = 0.0
            self.base_cmd.angular.z = 0.0

            self.base_vel_pub.publish(self.base_cmd)

            robot_orientation = Rotation.from_quat([self.state.orientation.x, self.state.orientation.y, self.state.orientation.z, self.state.orientation.w]).as_euler('xyz')

            walker_orientation = robot_orientation[2] + self.offset[5]

            walker_quaternion = Rotation.from_euler('xyz', [0, 0, walker_orientation]).as_quat()

            state_walker = ModelState()
            state_walker.model_name = "Walker";
            state_walker.reference_frame = "world";
            state_walker.pose.position.x = self.state.position.x + self.offset[0]
            state_walker.pose.position.y = self.state.position.y + self.offset[1]
            state_walker.pose.position.z = 0.1
            state_walker.pose.orientation.x = walker_quaternion[0]
            state_walker.pose.orientation.y = walker_quaternion[1]
            state_walker.pose.orientation.z = walker_quaternion[2]
            state_walker.pose.orientation.w = walker_quaternion[3]

            self.walkerPub(state_walker)

        else:

            self.base_cmd.linear.x = 0.0
            self.base_cmd.linear.y = 0.0
            self.base_cmd.linear.z = 0.0
            self.base_cmd.angular.x = 0.0
            self.base_cmd.angular.y = 0.0
            self.base_cmd.angular.z = 0.0

            self.base_vel_pub.publish(self.base_cmd)

            walker_orientation = patient_pose[3]

            walker_quaternion = Rotation.from_euler('xyz', [0, 0, walker_orientation]).as_quat()

            state_walker = ModelState()
            state_walker.model_name = "Walker";
            state_walker.reference_frame = "world";
            state_walker.pose.position.x = patient_pose[1] - 0.16*np.cos(patient_pose[3]+np.pi/2)
            state_walker.pose.position.y = patient_pose[0] - 0.16*np.sin(patient_pose[3]+np.pi/2)
            state_walker.pose.position.z = self.state.position.z + self.offset[2]
            state_walker.pose.orientation.x = walker_quaternion[0]
            state_walker.pose.orientation.y = walker_quaternion[1]
            state_walker.pose.orientation.z = walker_quaternion[2]
            state_walker.pose.orientation.w = walker_quaternion[3]

            self.walkerPub(state_walker)


    def base_callback(self, data):
        self.state = data.pose[-1]
        self.current_base = [self.state.position.x, self.state.position.y]
