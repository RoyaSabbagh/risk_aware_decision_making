from scipy import *
from scipy.linalg import norm, pinv
import numpy as np
from matplotlib import pyplot as plt
import math
import random
from shapely.geometry import Polygon, Point


def define_obstacles(env):
    ''' This function is to define line segments for each obstacle to be used in the trajectory optimization
    and also obstacle polygons to be used for sampling start and end points that are out of obstacles.'''

    obstacles = []
    r_patient = 0.1 # Margin to avoid obstacles


    for obj in env.objects:
        m_box = [math.tan(obj.conf.z), math.tan(obj.conf.z + math.pi / 2), math.tan(obj.conf.z), math.tan(obj.conf.z + math.pi / 2)]
        dobs = math.sqrt(obj.length**2+obj.width**2+2*r_patient)
        corners = np.asarray(obj.polygon.exterior.coords)
        b = [corners[0][1] - m_box[0] * corners[0][0], corners[1][1] - m_box[1] * corners[1][0],
                    corners[2][1] - m_box[2] * corners[2][0], corners[3][1] - m_box[3] * corners[3][0]]
        center_pose = [obj.conf.x, obj.conf.y]
        obstacles.append([m_box, b, center_pose, Polygon(corners)])

    for wall in env.walls:
        wall_c = [(wall[0][0]+wall[0][2])/2, (wall[0][1]+wall[0][3])/2]
        wall_d = [2*r_patient + np.sqrt((wall[0][0]-wall[0][2])**2+(wall[0][1]-wall[0][3])**2),2*r_patient + 0.6]
        wall_angle = np.arctan2((wall[0][1]-wall[0][3]), (wall[0][0]-wall[0][2])) + 0.001
        m_box = [math.tan(wall_angle), math.tan(wall_angle + math.pi / 2), math.tan(wall_angle), math.tan(wall_angle + math.pi / 2)]
        dobs = math.sqrt(wall_d[0]**2+wall_d[1]**2)
        corners = [[wall_c[0] - (wall_d[1]/2) * math.sin(wall_angle) - (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] + (wall_d[1]/2) * math.cos(wall_angle) - (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] + (wall_d[1]/2) * math.sin(wall_angle) - (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] - (wall_d[1]/2) * math.cos(wall_angle) - (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] + (wall_d[1]/2) * math.sin(wall_angle) + (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] - (wall_d[1]/2) * math.cos(wall_angle) + (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] - (wall_d[1]/2) * math.sin(wall_angle) + (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] + (wall_d[1]/2) * math.cos(wall_angle) + (wall_d[0]/2) * math.sin(wall_angle)]]
        b = [corners[0][1] - m_box[0] * corners[0][0], corners[1][1] - m_box[1] * corners[1][0],
                    corners[2][1] - m_box[2] * corners[2][0], corners[3][1] - m_box[3] * corners[3][0]]
        center_pose = [wall_c[0], wall_c[1]]
        obstacles.append([m_box, b, center_pose, Polygon(corners)])

    return obstacles

def sample_point(env, obj, obstacles):
    ''' This function samples a point around the target object. It can be a sitting zone for sittable furniture,
    a reaching zone for reachable objects such as bathroom sink, or just inside an area like main entrance door. '''

    x_min, x_max, y_min, y_max = [0, 10, 0, 10] # Sampling range
    found = False
    while not found:
        x = random.uniform(x_min,x_max)
        y = random.uniform(y_min,y_max)
        point = Point([x,y])

        # Check if the sampled point is in the sitting zone of the target object
        is_in_sample_zone = False
        if point.within(env.sample_zones[obj]):
            is_in_sample_zone = True

        # Check if the sampled point is out of all the obstacles in the room
        is_out_of_obstacle = True
        for obs in obstacles:
            if point.within(obs[3]):
                is_out_of_obstacle = False

        if is_in_sample_zone == True and is_out_of_obstacle == True:
            found = True
            point = [x,y, 0, 0, 0]

    return point

def is_near_sitting_object(state, env, obj):
    ''' This function determines if a point is near the sitting zone of a target object. '''

    is_near = False
    if obj in ['Bed_L', 'Bed_R', 'Toilet', 'Chair-Visitor']:
        if state.within(env.sample_zones[obj]):
            is_near = True
    return is_near

class RBF:

    def __init__(self, indim, numCenters, outdim, T):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = np.linspace(0, T, self.numCenters)
        self.beta = 5
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        # rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        # self.centers = [X[i,:] for i in rnd_idx]

        # print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        # print(G)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
        # print("W:")
        # print(self.W)

    def find(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

def find_corners(x, y, phi, width, length):
	''' This function finds corners of an object '''
	corners = []
	corners.append([x - (width/2) * np.sin(phi) - (length/2) * np.cos(phi), y + (width/2) * np.cos(phi) - (length/2) * np.sin(phi)])
	corners.append([x + (width/2) * np.sin(phi) - (length/2) * np.cos(phi), y - (width/2) * np.cos(phi) - (length/2) * np.sin(phi)])
	corners.append([x + (width/2) * np.sin(phi) + (length/2) * np.cos(phi), y - (width/2) * np.cos(phi) + (length/2) * np.sin(phi)])
	corners.append([x - (width/2) * np.sin(phi) + (length/2) * np.cos(phi), y + (width/2) * np.cos(phi) + (length/2) * np.sin(phi)])

	return corners
