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

#include <functional>

#include <ignition/math.hh>
#include "gazebo/physics/physics.hh"
#include "ActorPlugin.hh"
#include <ros/console.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "risk_aware_planning/patient_state.h"



using namespace gazebo;
GZ_REGISTER_MODEL_PLUGIN(ActorPlugin)

#define WALKING_ANIMATION "walking"
#define STAND_UP_ANIMATION "stand_up"
#define PI 3.14159265359
/////////////////////////////////////////////////
ActorPlugin::~ActorPlugin()
{
}

/////////////////////////////////////////////////
void ActorPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  this->sdf = _sdf;
  this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
  this->world = this->actor->GetWorld();

  this->connections.push_back(event::Events::ConnectWorldUpdateBegin(
          std::bind(&ActorPlugin::OnUpdate, this, std::placeholders::_1)));

  this->z_sit = { 0, -0.01, -0.04, -0.06, -0.13, -0.1, -0.07, 0.1, 0.14, 0.2, 0.22};

  this->Reset();
  this->counter = 0;

  // Read in the target weight
  if (_sdf->HasElement("target_weight"))
    this->targetWeight = _sdf->Get<double>("target_weight");
  else
    this->targetWeight = 1.15;

  // Read in the obstacle weight
  if (_sdf->HasElement("obstacle_weight"))
    this->obstacleWeight = _sdf->Get<double>("obstacle_weight");
  else
    this->obstacleWeight = 1.5;

  // Read in the animation factor (applied in the OnUpdate function).
  if (_sdf->HasElement("animation_factor"))
    this->animationFactor = _sdf->Get<double>("animation_factor");
  else
    this->animationFactor = 4.5;

  // Add our own name to models we should ignore when avoiding obstacles.
  this->ignoreModels.push_back(this->actor->GetName());

  // Read in the other obstacles to ignore
  if (_sdf->HasElement("ignore_obstacles"))
  {
    sdf::ElementPtr modelElem =
      _sdf->GetElement("ignore_obstacles")->GetElement("model");
    while (modelElem)
    {
      this->ignoreModels.push_back(modelElem->Get<std::string>());
      modelElem = modelElem->GetNextElement("model");
    }
  }
}

/////////////////////////////////////////////////
void ActorPlugin::Reset()
{
  this->velocity = 0.8;
  this->lastUpdate = 0;

  if (this->sdf && this->sdf->HasElement("target"))
    this->target = this->sdf->Get<ignition::math::Vector3d>("target");
  else
    this->target = ignition::math::Vector3d(0, -5, 1.2138);

  auto skelAnims = this->actor->SkeletonAnimations();
  if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end())
  {
    gzerr << "Skeleton animation " << WALKING_ANIMATION << " not found.\n";
  }
  else
  {
    // Create custom trajectory
    this->trajectoryInfo.reset(new physics::TrajectoryInfo());
    this->trajectoryInfo->type = WALKING_ANIMATION;
    this->trajectoryInfo->duration = 1.0;

    this->actor->SetCustomTrajectory(this->trajectoryInfo);
  }
}

/////////////////////////////////////////////////
void ActorPlugin::ChooseNewTarget()
{
  ignition::math::Vector3d newTarget(this->target);
  while ((newTarget - this->target).Length() < 1.0)
  {
    newTarget.X(ignition::math::Rand::DblUniform(-5, 1));
    newTarget.Y(ignition::math::Rand::DblUniform(-5, 5));

    for (unsigned int i = 0; i < this->world->ModelCount(); ++i)
    {
      double dist = (this->world->ModelByIndex(i)->WorldPose().Pos()
          - newTarget).Length();
      if (dist < 2.0)
      {
        newTarget = this->target;
        break;
      }
    }
  }
  this->target = newTarget;
}

/////////////////////////////////////////////////
void ActorPlugin::HandleObstacles(ignition::math::Vector3d &_pos)
{
  for (unsigned int i = 0; i < this->world->ModelCount(); ++i)
  {
    physics::ModelPtr model = this->world->ModelByIndex(i);
    if (std::find(this->ignoreModels.begin(), this->ignoreModels.end(),
          model->GetName()) == this->ignoreModels.end())
    {
      ignition::math::Vector3d offset = model->WorldPose().Pos() -
        this->actor->WorldPose().Pos();
      double modelDist = offset.Length();
      if (modelDist < 4.0)
      {
        double invModelDist = this->obstacleWeight / modelDist;
        offset.Normalize();
        offset *= invModelDist;
        _pos -= offset;
      }
    }
  }
}

void ActorPlugin::PlayWithAnimationName(const std::string &_animationName,
                                  const bool &_completeScript,
                                  const boost::optional<unsigned int> _id)
{
  if (this->actor->skelAnimation[_animationName])
  {
    this->actor->startTime = 0.0;
    this->actor->scriptLength = 0.0;
    bool found = false;
    // ROS_INFO("%zd ", this->actor->trajInfo.size());
    for (unsigned int i = 0; i < this->actor->trajInfo.size(); ++i)
    {
      // TrajectoryInfo t = this->actor->trajInfo[i];
      this->actor->scriptLength += this->actor->trajInfo[i].duration;
      bool cond = false;
      // ROS_INFO("%s\n", _animationName.c_str());
      if (_id)
      {
        cond = (this->actor->trajInfo[i].type == _animationName);
      }
      else
      {
        cond = (this->actor->trajInfo[i].type == _animationName);
      }

      if (cond)
      {
        this->actor->startTime = this->actor->scriptLength - this->actor->trajInfo[i].duration;
        found = cond;
        if (!_completeScript)
        {
          break;
        }
      }
    }

    this->actor->scriptLength -= this->actor->startTime;
    // if (_id && !found)
    // {
    //   gzerr << "Invalid id" << std::endl;
    //   return;
    // }
    this->actor->Play();
  }
  else
  {
    gzerr << _animationName << " not found" << std::endl;
    return;
  }
}



bool ActorPlugin::play(const Animation &animation, const bool complete)
{
    // updateMap(targets);

    physics::Actor::SkeletonAnimation_M skel_m=actor->SkeletonAnimations();
    std::string name=animation.name;
    // if (!name.empty() && !skel_m[name])
    // {
    //     gzerr << "Animation not found";
    //     return false;
    // }
    // if (name.empty())
    // {
    //     gzerr <<"Playing whole script";
    //     actor->Play();
    // }
    // else
    // {
    //     if (complete)
    //     {
    //         gzerr<<"Playing script starting from "<<name;
    //     }
    //     else
    //     {
    //         gzerr<<"Playing "<<name;
    //     }
    // }
    if (animation.id>=0)
    {
        unsigned int id=animation.id;
        this->PlayWithAnimationName(name,complete,id);
    }
    else
    {

        this->PlayWithAnimationName(name,complete,-1);
    }

    return true;
}

void ActorPlugin::patientPoseCallback(risk_aware_planning::patient_state msg)
{
  if (msg.activity=="walking")
  {
      std::cout << "Walking..." << std::endl;
      this->trajectoryInfo->type = WALKING_ANIMATION;
      this->trajectoryInfo->duration = 1.0;
      // ROS_INFO("%s\n", msg.activity.c_str());

      ignition::math::Pose3d pose = this->actor->WorldPose();
      ignition::math::Vector3d rpy = pose.Rot().Euler();
      ignition::math::Angle phi = msg.pose.orientation.z;

      pose.Rot() = ignition::math::Quaterniond(0.5*PI, 0, phi.Radian());
      this->actor->SetWorldPose(pose, false, false);

      pose.Pos().X(msg.pose.position.x);
      pose.Pos().Y(msg.pose.position.y);
      pose.Pos().Z(0.7);
      // pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+yaw.Radian());
      double distanceTraveled = (pose.Pos() - this->actor->WorldPose().Pos()).Length();

      this->actor->SetWorldPose(pose, false, false);
      this->actor->SetScriptTime(this->actor->ScriptTime() +
        (distanceTraveled * this->animationFactor));
      this->lastUpdate = this->world->SimTime();
      this->counter = 0;
  }
  else if (msg.activity=="sit_to_stand")
  {
    std::cout << "Standing..." << this->counter<< std::endl;
    ignition::math::Pose3d pose = this->actor->WorldPose();
    ignition::math::Vector3d rpy = pose.Rot().Euler();
    ignition::math::Angle phi = msg.pose.orientation.z;

    pose.Pos().X(msg.pose.position.x-0.15);
    pose.Pos().Y(msg.pose.position.y+0.1);
    pose.Pos().Z(msg.pose.position.z);

    pose.Rot() = ignition::math::Quaterniond(0.5*PI, 0, phi.Radian());
    this->actor->SetWorldPose(pose, false, false);

    this->trajectoryInfo->type = STAND_UP_ANIMATION;
    this->actor->SetScriptTime(this->counter * 0.11);
    this->counter++;
  }
  else if (msg.activity=="stand_to_sit")
  {
    std::cout << "Sitting..." << this->counter<< std::endl;
    ignition::math::Pose3d pose = this->actor->WorldPose();
    ignition::math::Vector3d rpy = pose.Rot().Euler();
    ignition::math::Angle phi = msg.pose.orientation.z;

    pose.Pos().X(msg.pose.position.x-0.1);
    pose.Pos().Y(msg.pose.position.y);
    pose.Pos().Z(msg.pose.position.z);

    pose.Rot() = ignition::math::Quaterniond(0.5*PI, 0, phi.Radian());
    this->actor->SetWorldPose(pose, false, false);

    this->trajectoryInfo->type = STAND_UP_ANIMATION;
    this->actor->SetScriptTime((50-this->counter) * 0.11);
    this->counter++;
  }
  else
  {
    gzerr << "Animation not found..." << std::endl;
  }

  ROS_INFO("*****************");
}

/////////////////////////////////////////////////
void ActorPlugin::OnUpdate(const common::UpdateInfo &_info)
{

  if (!ros::isInitialized())
  {
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "patient_traj",
        ros::init_options::NoSigintHandler);
  }
  Animation anim;
  anim.name = "stand_up";
  anim.id = -1;
  this->play(anim, 0);
  ros::NodeHandle n;
  ros::AsyncSpinner spinner(50);
  pose_sub_ = nh_.subscribe("patient_pose", 1, &ActorPlugin::patientPoseCallback, this);

  // ros::spin();
  // Time delta
  double dt = (_info.simTime - this->lastUpdate).Double();

  ignition::math::Pose3d pose = this->actor->WorldPose();
  ignition::math::Vector3d pos = this->target - pose.Pos();
  ignition::math::Vector3d rpy = pose.Rot().Euler();

  double distance = pos.Length();

  // Choose a new target position if the actor has reached its current
  // target.
  if (distance < 0.3)
  {
    this->ChooseNewTarget();
    pos = this->target - pose.Pos();
  }
  // ROS_INFO("target-x: %f", this->target.X());
  // ROS_INFO("target-y: %f", this->target.Y());
  // ROS_INFO("target-z: %f", this->target.Z());
  //
  // ROS_INFO("pose.Pos()-x: %f", pose.Pos().X());
  // ROS_INFO("pose.Pos()-y: %f", pose.Pos().Y());
  // ROS_INFO("pose.Pos()-z: %f", pose.Pos().Z());
  //
  // ROS_INFO("1pos-x: %f", pos.X());
  // ROS_INFO("pos-y: %f", pos.Y());
  // ROS_INFO("pos-z: %f", pos.Z());
  // Normalize the direction vector, and apply the target weight
  pos = pos.Normalize() * this->targetWeight;
  // ROS_INFO("2pos-x: %f", pos.X());
  // ROS_INFO("pos-y: %f", pos.Y());
  // ROS_INFO("pos-z: %f", pos.Z());
  // Adjust the direction vector by avoiding obstacles
  this->HandleObstacles(pos);
  // ROS_INFO("3pos-x: %f", pos.X());
  // ROS_INFO("pos-y: %f", pos.Y());
  // ROS_INFO("pos-z: %f", pos.Z());
  // Compute the yaw orientation
  ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
  yaw.Normalize();

  // Rotate in place, instead of jumping.
  if (std::abs(yaw.Radian()) > IGN_DTOR(10))
  {
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+
        yaw.Radian()*0.001);
  }
  else
  {
    // ROS_INFO("pos-x: %f", pos.X());
    // ROS_INFO("pos-y: %f", pos.Y());
    // ROS_INFO("pos-z: %f", pos.Z());
    pose.Pos() += pos * this->velocity * dt;
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+yaw.Radian());
  }



  // Make sure the actor stays within bounds
  // pose.Pos().X(std::max(-4.0, std::min(1.0, pose.Pos().X())));
  // pose.Pos().Y(std::max(-5.0, std::min(5.0, pose.Pos().Y())));
  // pose.Pos().Z(1.05);
  // pose.Pos() = pose.Pos() + this->velocity * dt;
  // pose.Pos().Z(this->fixed_actor_height);

  // Distance traveled is used to coordinate motion with the walking
  // animation
  // double distanceTraveled = (pose.Pos() -
  //     this->actor->WorldPose().Pos()).Length();
  //
  // this->actor->SetWorldPose(pose, false, false);
  // this->actor->SetScriptTime(this->actor->ScriptTime() +
  //   (distanceTraveled * this->animationFactor));
  // this->lastUpdate = _info.simTime;
}
