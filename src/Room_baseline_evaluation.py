#!/usr/bin/env python


from Fall_risk_assesment import Environment_Image, FallRiskAssesment
import numpy as np
# import random
from Optimization import OptPath_patient, OptPath_patient2, nominal_traj, OptTraj
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.cbook as cbook
import matplotlib.image as image
from scipy import *
from utility import define_obstacles, sample_point, is_near_sitting_object, RBF, find_corners
import random
from shapely.geometry import Polygon, Point
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF
import copy
import pickle



if __name__ == '__main__':

    # ***************************************** Inputs ******************************************************

    # Set file addresses
    input_type = "image"
    path = "/home/roya/catkin_ws/src/risk_aware_planning"
    design_name = "Room-2-Inboard-Footwall"
    day_night = "night"
    plots = False

    # Setup parameters for evaluation
    unit_size_m = 0.2
    num_rows = 50
    num_cols = 50

    library_file = "{0}/Object_Library.csv".format(path) # Put the object library file address here.
    image_file = "{0}/Room_Designs/{1}_{2}.png".format(path, design_name, day_night) # Put the image file address here.

    background_filename = "{0}/Room_Designs/{1}_objects_rotated.png".format(path, design_name)

    # ************************************** Setup environment *************************************************

    print("Environment Setup...")
    if input_type == "image":
        env = Environment_Image(image_file, library_file, unit_size_m, num_rows, num_cols) # Basically, reads an input image and setups the room environment properties for fall risk assessment
    elif input_image == "generated":
        env = Environment_Generated(unit_size_m, num_rows, num_cols)

    # ************************************ Baseline evaluation **************************************************

    print("Baseline Evaluation...")
    fallRisk = FallRiskAssesment(env) # Initial FallRiskAssesment class
    fallRisk.update(False, plots) # Find scores for each baseline factor and baseline evaluation
    # np.save("data/{0}_baseline_evaluation.npy".format(design_name), fallRisk.scores)
    pickle_baseline_evaluation = open("data/{0}/baseline_evaluation.pickle".format(design_name),"wb")
    pickle.dump(fallRisk, pickle_baseline_evaluation)
    pickle_baseline_evaluation.close()

    pickle_env = open("data/{0}/env.pickle".format(design_name),"wb")
    pickle.dump(env, pickle_env)
    pickle_env.close()
