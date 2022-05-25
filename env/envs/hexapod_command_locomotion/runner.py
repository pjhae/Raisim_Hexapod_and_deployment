from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import hexapod_command_locomotion
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher, Command
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "hexapod_command_locomotion"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# rewards dict for tb logging
reward_names = list(map(str, cfg['environment']['reward'].keys()))
reward_names.append('reward_sum')

# for custom command
command = Command(cfg)
n_steps_command = cfg['environment']['command']['dt'] / cfg['environment']['control_dt']
command_dict = {0:'Go Forward', 1:'Turn Left', 2:'Turn Right', 3:'Go Backward', 4:'Go Left', 5:'Go Right'} # should be updated manually when changed!!

# create environment from the configuration file
env = VecEnv(hexapod_command_locomotion.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=16, # or, just 4
              learning_rate=3e-5, # can be smaller
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              # can adjust gamma, lambda values too
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

# for rewards logger
recent_reward_means = dict()
for name in reward_names:
    recent_reward_means[name] = []

# for info logger
recent_dones = 0.
recent_elapsed = 0.


for update in range(1000000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):
            with torch.no_grad():
                frame_start = time.time()

                # give command
                if step % n_steps_command == 0:
                    sample_command = command.sample_evaluate()
                    env.setCommand(sample_command)
                    print(command_dict[sample_command[0]])

                obs = env.observe(False)
                action_ll = loaded_graph(torch.from_numpy(obs).cpu())
                reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        # env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            # give command
            if step % n_steps_command == 0:
                sample_command = command.sample_train()
                env.setCommand(sample_command)

            obs = env.observe()
            action = ppo.observe(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)

            last_rewards = env.getRewardInfo()
            for name in reward_names:
                recent_reward_means[name].append(np.mean(list(r[name] for r in last_rewards)))

    # take st step to get value obs
    obs = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(18)*0.2).to(device))

    # reward logging
    if update % 10 == 0:
        ppo.log_reward(reward_names, recent_reward_means, update)
        for name in reward_names:
            recent_reward_means[name] = []

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    # info logging
    recent_dones += average_dones
    recent_elapsed += (end-start)
    if update % 10 == 0:
        ppo.log_info(recent_dones/10., recent_elapsed/10., update)
        recent_dones = 0.
        recent_elapsed = 0.

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
