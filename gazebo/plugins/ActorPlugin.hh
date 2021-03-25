/*
 * Copyright (C) 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#ifndef GAZEBO_PLUGINS_ACTORPLUGIN_HH_
#define GAZEBO_PLUGINS_ACTORPLUGIN_HH_

#include <string>
#include <vector>
#include <thread>
#include "ros/ros.h"
#include "ros/rate.h"
#include <ros/node_handle.h>

#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include <geometry_msgs/PoseStamped.h>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"
#include <gazebo/gazebo.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/Actor.hh>
#include "gazebo/common/Animation.hh"
#include "risk_aware_planning/patient_state.h"

struct Animation
{
std::string name="";     /* animation name */
int id=-1;          /* animation id */
};


namespace gazebo
{
  class GZ_PLUGIN_VISIBLE ActorPlugin : public ModelPlugin
  {
    /// \brief Constructor
    public:

    ActorPlugin(): ModelPlugin(),
                  rate_(50) {}

    virtual ~ActorPlugin();
    /// \brief Load the actor plugin.
    /// \param[in] _model Pointer to the parent model.
    /// \param[in] _sdf Pointer to the plugin's SDF elements.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

    // Documentation Inherited.
    public: virtual void Reset();

    /// \brief Function that is called every update cycle.
    /// \param[in] _info Timing information
    private: void OnUpdate(const common::UpdateInfo &_info);

    /// \brief Helper function to choose a new target location
    private: void ChooseNewTarget();

    /// \brief Helper function to avoid obstacles. This implements a very
    /// simple vector-field algorithm.
    /// \param[in] _pos Direction vector that should be adjusted according
    /// to nearby obstacles.
    private: void HandleObstacles(ignition::math::Vector3d &_pos);

    /// \brief Pointer to the parent actor.
    private: physics::ActorPtr actor;

    /// \brief Pointer to the world, for convenience.
    private: physics::WorldPtr world;

    /// \brief Pointer to the sdf element.
    private: sdf::ElementPtr sdf;

    /// \brief Velocity of the actor
    private: ignition::math::Vector3d velocity;

    /// \brief List of connections
    private: std::vector<event::ConnectionPtr> connections;

    /// \brief Current target location
    private: ignition::math::Vector3d target;

    /// \brief Target location weight (used for vector field)
    private: double targetWeight = 1.0;

    /// \brief Obstacle weight (used for vector field)
    private: double obstacleWeight = 1.0;

    /// \brief Time scaling factor. Used to coordinate translational motion
    /// with the actor's walking animation.
    private: double animationFactor = 1.0;

    /// \brief Time of the last update.
    private: common::Time lastUpdate;

    /// \brief List of models to ignore. Used for vector field
    private: std::vector<std::string> ignoreModels;

    /// \brief Custom trajectory info.
    private: physics::TrajectoryInfoPtr trajectoryInfo;

    public: double counter = 1.0;

    public: std::vector<double> z_sit;

    /// \brief A ROS subscriber
    private: ros::Subscriber pose_sub_;

    private: ros::Rate rate_;
    /// \brief A node use for ROS transport
    private: ros::NodeHandle nh_;

    private: geometry_msgs::PoseStamped patientPose_;

    void patientPoseCallback(risk_aware_planning::patient_state pose);

    /// \brief topic name
    private: std::string topic_name_;
    //
    //
    //
    //
    // /// \brief A ROS callbackqueue that helps process messages
    // private: ros::CallbackQueue rosQueue_;
    //
    // private: void QueueThread();
    //
    // /// \brief A thread the keeps running the rosQueue
    // private: std::thread rosQueueThread_;
    /**
     * Play specified animation.
     * @param animation in the form (name, id). If not specified, all animations are played according to their id.
     * The id has to be specified if the animation is played several times during the script.
     * @param complete if true, the whole script is played starting from the specified animation.
     * @return returns true / false on success / failure.
     */
    public: virtual bool play(const Animation &animation, const bool complete);

    public: virtual void PlayWithAnimationName(const std::string &_animationName,
                                      const bool &_completeScript,
                                      const boost::optional<unsigned int> _id);


  };
}
#endif
