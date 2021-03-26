# risk_aware_decision_making

This is a ROS package that represents a risk aware decision making framework for assistive robots. 
It is authored by [Roya Sabbagh Novin](https://sites.google.com/view/roya-sn), [Amir Yazdani](https://amir-yazdani.github.io/), [Andrew Merryweather](https://mech.utah.edu/faculty/andrew-merryweather/), and [Tucker Hermans](https://robot-learning.cs.utah.edu/thermans) from Utah Robotics Center, University of Utah.

More information about the paper and the method can be found in the [paper website](https://sites.google.com/view/risk_aware_decision_making) and a PDF version is available at the [arXiv pape](https://arxive.org/abs/2010.08124).
        
        
   [![Sammary](http://img.youtube.com/vi/WmYb1sxsIjg/0.jpg)](https://www.youtube.com/watch?v=WmYb1sxsIjg "Summary")



### Features
- **Functionality**:
  - **Risk-aware decision making for an assistive robot** 
    - Find the best intervention plan
    - Considering risk of fall for the patient
    - leveraging CVaR risk metric
  - **Human motion/intention prediction and fall score assessment**
    - Generated human motion data in hospital room
    - Learned GPs for human motion
    - Predict probability distributions over intention and trajectory
    - Using developed fall score assessment model ([Github repository for fall risk evaluation](https://github.com/RoyaSabbagh/fall_risk_evaluation))
  - **Simulation**
    - Gazebo visualization for simulation results
    - Provided Gazebo actor plugin to simulate patient motion
    - Including urdf for common objects in a hospital room
- **Input**: Robot intial position, object type, object initial position and orientation, patient initial position and orientation, layot of the environment
- **Output**: Human motion prediction, best intervention plan for robot, executed path for robot
- **Operating System**: Ubuntu (16.04), ROS kinetic

### Dependencies 

  - **Gazebo 8+**
    Install Gazebo any version above 8 should work. This package has been tested with Gazebo 9.12.0 for ubuntu.
  - **youBot packages**
    Install youbot_simulation and youbot_description packages. ([link](http://www.youbot-store.com/wiki/index.php/Gazebo_simulation))


### Installation
1. Make sure you have ROS-kinetic installed and your catkin workspace is generated
2. Clone the package in your catkin workspace
3. Make/build your catkin workspace
4. Enjoy!


### How to set up and run
1. Set a simulation example in "main_simulation.py" or choose one from the available ones"
3. Run "launch_gazebo.launch"

### Citation
Please cite these papers in your publications if it helps your research.

        @article{novin2020risk,
        title={Risk-Aware Decision Making in Service Robots to Minimize Risk of Patient Falls in Hospitals},
        author={Sabbagh Novin, Roya and Yazdani, Amir and Merryweather, Andrew and Hermans, Tucker},
        journal={arXiv preprint arXiv:2010.08124},
        year={2020}
        }

Links to the papers:

- [Risk-Aware Decision Making in Service Robots to Minimize Risk of Patient Falls in Hospitals](https://arxive.org/abs/2010.08124)


