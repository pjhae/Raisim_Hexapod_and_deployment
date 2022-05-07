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

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget18_.setZero(nJoints_);

    /// this is nominal configuration of phantom
    gc_init_ << 0, 0, 0.10, 1.0, 0.0, 0.0, 0.0;
    for(int i = 7; i < gcDim_ ; i ++ ) gc_init_ << 0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(3.125); // determined by ST
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.095); // determined by ST
    phantom_->setPdGains(jointPgain, jointDgain);
    phantom_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDim_ = 6*nJoints_+3; // joint state history and action history + command
    obDouble_.setZero(obDim_); 
    jnt_hist_.setZero(3*nJoints_); act_hist_.setZero(3*nJoints_);

    /// action scaling 
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

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
    initRandomization_ = cfg["randomization"]["initialization"].template As<bool>();

    /// for custom command
    command_.setZero(3); // 3 elements : xAxisVel, yAxisVel, yawRate (in order)

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(phantom_);
    }
  }

  void init() final {  }

  void reset() final {
    phantom_->setState(gc_init_, gv_init_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget18_.setZero(nJoints_);
    obDouble_.setZero(obDim_); jnt_hist_.setZero(3*nJoints_); act_hist_.setZero(3*nJoints_);

    /// Domain Randomization
    if (massRandomization_) { /// varying mass
        auto &mass_values = phantom_->getMass();
        for (int i = 0; i < nJoints_ + 1; i++) {
            mass_values[i] = originalMassValues_[i] * (1 - 0.15 + 2 * 0.15 * dist_(mersenne_engine_));
        }
        phantom_->updateMassInfo();
    }
    
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget18_ = action.cast<double>();
    pTarget18_ = pTarget18_.cwiseProduct(actionStd_);
    pTarget18_ += actionMean_;
    pTarget18_ = pTarget18_.cwiseMin(1.2).cwiseMax(-1.2);
    pTarget_.tail(nJoints_) = pTarget18_;

    phantom_->setPdTarget(pTarget_, vTarget_);

    for (int i=0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    
    /// compute rewards 
    
    Eigen::Vector3d com_lin_vel, coeff_lin_vel_err;
    com_lin_vel << command_.head(2), 0.0;
    coeff_lin_vel_err << 10.0, 10.0, 3.0;
    
    double vel_error = (com_lin_vel-bodyLinearVel_).cwiseProduct(coeff_lin_vel_err).norm();
    double yaw_rate_error = (command_[2]-bodyAngularVel_[2])*(command_[2]-bodyAngularVel_[2]);

    rewards_.record("linVelErr", 1.0/(exp(vel_error)+2+exp(-vel_error)));
    rewards_.record("yawRateErr", 1.0/(exp(yaw_rate_error)+2+exp(-yaw_rate_error)));
    // rewards_.record("torque", sqrt(phantom_->getGeneralizedForce().squaredNorm()));
    // rewards_.record("headingAng", sqrt(bodyEuler_.head(2).squaredNorm()));

    return rewards_.sum();
  }

  void updateObservation() {
    phantom_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    raisim::Vec<3> euler;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    raisim::quatToEulerVec(quat.ptr(), euler.ptr());
    bodyEuler_ = euler.e();

    jnt_hist_ << jnt_hist_.tail(2*nJoints_), gc_.tail(nJoints_);
    act_hist_ << act_hist_.tail(2*nJoints_), pTarget18_;
    obDouble_ << jnt_hist_, act_hist_, command_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: phantom_->getContacts()) {
      if (contact.skip())
        return true;
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }

    /// if the center is too low
    if (gc_[2] < 0.03) return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {
    /// can add vanishing coeff of some reward term... (like scale1 *= 0.997)
  };

  void setCommand(Eigen::Ref<EigenVec> command) {
    /// translationVel, rotationVel, direction (in order)
    for (int i = 0; i < 3; i++){
        command_[i] = command[i];
    }
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* phantom_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget18_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// for quat of gc_ => rpy
  Eigen::Vector3d bodyEuler_;

  /// for history input
  Eigen::VectorXd jnt_hist_, act_hist_;

  /// for randomization
  bool massRandomization_, initRandomization_;
  std::vector<double> originalMassValues_;

  /// for random generation
  std::random_device rnd_device;
  std::mt19937 mersenne_engine_ {rnd_device()};
  std::uniform_real_distribution<double> dist_;

  /// for custom command
  Eigen::VectorXd command_;
};
}

