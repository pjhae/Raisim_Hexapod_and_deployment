//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    phantom_ = world_->addArticulatedSystem(resourceDir_+"/hexapod/urdf/hexapod.urdf");
    phantom_->setName("ASSY_phantom");
    phantom_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = phantom_->getGeneralizedCoordinateDim(); // 25 (xyz + quat +joints)
    gvDim_ = phantom_->getDOF(); // 24 (xyz + euler + joints)
    nJoints_ = gvDim_ - 6;
    
    /// action clipping for joint limit
    joint_low_limit_.setZero(nJoints_); joint_high_limit_.setZero(nJoints_);
    joint_low_limit_ << -1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0,
                        -1.0, -1.0, 0.0;
    joint_high_limit_ << 1.0, 0.0, 1.0,
                         1.0, 0.0, 1.0,
                         1.0, 0.0, 1.0,
                         1.0, 0.0, 1.0,
                         1.0, 0.0, 1.0,
                         1.0, 0.0, 1.0;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget18_.setZero(nJoints_);

    /// this is nominal configuration of phantom
    gc_init_.segment(0, 3) << 0.0, 0.0, 0.1; // body x y z
    gc_init_.segment(3, 4) << 1.0, 0.0, 0.0, 0.0; // body quat
    // gc_init_.tail(nJoints_) = (joint_low_limit_ + joint_high_limit_) / 2.0; // middle point of joint limits
    gc_init_.tail(nJoints_).setZero();

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(2.96577); // determined by ST
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.09308); // determined by ST
    phantom_->setPdGains(jointPgain, jointDgain);
    phantom_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// for custom command
    // command_.setZero(3); // for vector of command
    num_command_classes_ = cfg["command"]["num_classes"].template As<int>();
    one_hot_command_.setZero(num_command_classes_);

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDim_ = 3*nJoints_ + num_command_classes_; // joint state history history + command
    obDouble_.setZero(obDim_); 
    jnt_hist_.setZero(3*nJoints_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile(cfg["reward"]);

    /// indices of links that should make contact with ground
    footIndices_.insert(phantom_->getBodyIdx("LF_foot"));
    footIndices_.insert(phantom_->getBodyIdx("LM_foot"));
    footIndices_.insert(phantom_->getBodyIdx("LB_foot"));
    footIndices_.insert(phantom_->getBodyIdx("RF_foot"));
    footIndices_.insert(phantom_->getBodyIdx("RM_foot"));
    footIndices_.insert(phantom_->getBodyIdx("RB_foot"));
    
    /// save original mass values of phantom for DR
    originalMassValues_ = phantom_->getMass();

    /// randomize or not
    massRandomization_ = cfg["randomization"]["mass"].template As<bool>();
    
    /// vanishing rewards coeff
    torque_cost_scale_ = 1.0;
    
    /// for custom command
    // initialize containers as init position (because there is no command at VecEnv initialization time !!)
    gc_ = gc_init_; setCommand(-1); // -1 command means nothing (all zero vector)

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(phantom_);
    }
  }

  void init() final {  }

  void reset() final {
  
    /// Domain Randomization
    // dist_(mersenne_engine_) : 0.0 ~ 1.0 unif dist.
    
    if (massRandomization_) { // varying mass
        auto &mass_values = phantom_->getMass();
        for (int i = 0; i < nJoints_+1; i++) {
            mass_values[i] = originalMassValues_[i] * (1 - 0.05 + 2 * 0.05 * dist_(mersenne_engine_)); // +- 5%
        }
        phantom_->updateMassInfo();
    }
    
    phantom_->setState(gc_init_, gv_init_); // if rand init is completed, erase this line
    
    pTarget_.setZero(); vTarget_.setZero(); pTarget18_.setZero();
    obDouble_.setZero(); jnt_hist_.setZero(); // each states history buffer stores three timesteps 
    
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget18_ = action.cast<double>();
    pTarget18_ = pTarget18_.cwiseProduct(actionStd_);
    pTarget18_ += actionMean_;
    pTarget18_ = pTarget18_.cwiseMax(joint_low_limit_).cwiseMin(joint_high_limit_);
    pTarget_.tail(nJoints_) = pTarget18_;

    phantom_->setPdTarget(pTarget_, vTarget_);

    for (int i=0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    
    /// compute rewards 
    switch(command_) {
        case GO_FORWARD:
            rewards_.record("linearVel", relevant_lin_vel_[0]);
            break;
        case TURN_LEFT:
            rewards_.record("angularVel", gv_[5]); // +vel_Rz
            break;
        case TURN_RIGHT:
            rewards_.record("angularVel", -gv_[5]); // -vel_Rz
            break;
        case GO_BACKWARD:
            rewards_.record("linearVel", -relevant_lin_vel_[0]);
            break;
        case GO_LEFT:
            rewards_.record("lateralVel", -relevant_lin_vel_[1]);
            break;
        case GO_RIGHT:
            rewards_.record("lateralVel", relevant_lin_vel_[1]);
            break;
    }
    
    rewards_.record("torque", torque_cost_scale_ * phantom_->getGeneralizedForce().squaredNorm());

    return rewards_.sum();
  }

  void updateObservation() {
    phantom_->getState(gc_, gv_);
    
    /*raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToEulerVec(quat.ptr(), euler.ptr());
    bodyEuler_ = euler.e();*/
    
    relevant_xy_ = command_rot_mat_.transpose() * (gc_.head(2) - start_xy_);
    relevant_lin_vel_ = command_rot_mat_.transpose() * gv_.head(2);

    jnt_hist_ << jnt_hist_.tail(obDim_-nJoints_-num_command_classes_), gc_.tail(nJoints_);
    
    obDouble_ << jnt_hist_, one_hot_command_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_); // terminal cost

    /// if the contact body is not feet
    for (auto& contact: phantom_->getContacts()) {
      if (contact.skip())
        return true;
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }
    
    /// when hexapod in unhealthy states, terminates episode

    // if the center is too low - common terminal condition
    if (gc_[2] < 0.03) return true;
    
    // if tilted too much
    switch(command_) {
        case GO_FORWARD:
            if (abs(relevant_xy_[1]) > 0.3) return true; // when one walks too far to the side
            break;
        case TURN_LEFT:
            if (relevant_xy_.norm() > 0.2) return true; // when one turns too far from the center
            break;
        case TURN_RIGHT:
            if (relevant_xy_.norm() > 0.2) return true; // when one turns too far from the center
            break;
        case GO_BACKWARD:
            if (abs(relevant_xy_[1]) > 0.3) return true; // when one walks too far to the side
            break;
        case GO_LEFT:
            if (abs(relevant_xy_[0]) > 0.3) return true; // when one walks too far to the side
            break;
        case GO_RIGHT:
            if (abs(relevant_xy_[0]) > 0.3) return true; // when one walks too far to the side
            break;
    }

    terminalReward = 0.f; // no cost because env didn't terminated
    
    return false;
  }

  void curriculumUpdate() {
    /// can add vanishing coeff of some reward term... (like scale1 *= 0.997)
    // called for each training 'iteration' (see runner.py)
    torque_cost_scale_ *= 0.999;
  };

  void setCommand(int command) {
    /// only '-1(special), and 0 ~ num_classes(cfg)' range commands are given
    // 0: go forward, 1: turn left, 2: turn right
    // 3: go backward, 4: go left, 5: go right
    command_ = command;
    
    /// -1 comm means env is reset
    if (command_ == -1) {
        start_xy_ = gc_init_.head(2); // current x, y position
        relevant_xy_.setZero(); // reset relevant position
        
        command_rot_mat_ << 1.0, 0.0,
                            0.0, 1.0;
        relevant_lin_vel_ = gv_.head(2);
        
        one_hot_command_.setZero();
        obDouble_.tail(num_command_classes_).setZero();
        
        return;
    }
    
    /// save start position and orientation for rewards and terminal states
    start_xy_ = gc_.head(2); // current x, y position
    relevant_xy_.setZero(); // reset relevant position
    
    double norm = std::sqrt(gc_[4]*gc_[4] + gc_[5]*gc_[5] + gc_[6]*gc_[6]);
    double yaw;
    if (abs(norm) < 1e-12) yaw = 0.0;
    else yaw = gc_[6] * std::acos(std::min(gc_[3],1.0)) * 2.0 / norm;
    command_rot_mat_ << std::cos(yaw), -std::sin(yaw),
                        std::sin(yaw), std::cos(yaw); // 2d rotation mtx for yaw on xy plane
    relevant_lin_vel_ = command_rot_mat_.transpose() * gv_.head(2);
    
    /// command one-hot encoding for input of neural net
    one_hot_command_.setZero();
    one_hot_command_(command_) = 1.0;
    
    /// update observation
    obDouble_.tail(num_command_classes_) << one_hot_command_;
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* phantom_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget18_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  std::set<size_t> footIndices_;

  /// for history input
  Eigen::VectorXd jnt_hist_;

  /// for randomization
  bool massRandomization_, initRandomization_;
  std::vector<double> originalMassValues_;

  /// for random generation
  std::random_device rnd_device;
  std::mt19937 mersenne_engine_ {rnd_device()};
  std::uniform_real_distribution<double> dist_;
  
  /// vanishing scale of some rewards coeff
  double torque_cost_scale_;
  
  /// action clipping for joint limit
  Eigen::VectorXd joint_low_limit_, joint_high_limit_;
  
  /// commands list
  enum Command_lists {
    GO_FORWARD, TURN_LEFT, TURN_RIGHT,
    GO_BACKWARD, GO_LEFT, GO_RIGHT
  };

  /// for custom command
  int command_, num_command_classes_;
  Eigen::VectorXd one_hot_command_;
  
  /// for terminal states when command transited, save new original info
  Eigen::Vector2d start_xy_, relevant_xy_, relevant_lin_vel_;
  Eigen::Matrix2d command_rot_mat_;
};
}

