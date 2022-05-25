# Raisim_Hexapod
## 2021 SNU URP project winner
### · Implementation of walking of a Hexapod robot using RL
### · Sim-to-real gap minimization through simulation parameter tuning.

## Requirments

Installation : RAISIM [[link](https://raisim.com/sections/Installation.html)] + (linux)


  1. Move to

    YOUR_PATH/raisimLib/raisimGymTorch/raisimGymTorch/

  2. Clone the repository in

    YOUR_PATH/raisimLib/raisimGymTorch/raisimGymTorch/

  3. Move [hexapod] directory to

    YOUR_PATH/raisimLib/rsc/

  4. set up environment

    cd YOUR_PATH/raisimLib/raisimGymTorch/
    python setup.py develop
  
  5. Train/Test

    cd YOUR_PATH/raisimLib/raisimGymTorch/env/envs/hexapod_command_locomotion
    (Training) python runner.py
    (Test) python tester.py -w [model_PATH]

  6. (optional) Deployment

    cd YOUR_PATH/raisimLib/raisimGymTorch/env/envs/hexapod_command_locomotion
    python deploy.py -w [model_PATH]



## 0. Setup

**Robot Hardware** : *PhantomX-MK3* [[INFO](https://www.trossenrobotics.com/Quadruped-Robot-Hexapod-Robot-Kits.aspx)]

**Communication** : *Dynamixel SDK(#18 of AX-12A) & PC* by **U2D2** [[INFO](https://www.robotis.com/shop/item.php?it_id=902-0132-000)]


![image](https://user-images.githubusercontent.com/74540268/170243754-6a16f510-fda8-4b47-a6e6-099610fb7e5e.png)

*Solidworks to URDF* : All links and joints are manually reverse engineered using assembly file from GRABCAD [[link](https://grabcad.com/library)]




## 1. Motor parameter tuning Algorithm

*ASSUMTION 1* : The main cause of the Sim-to-Real gap is the **inaccuracy of motor modeling** and **the latency of the timestep.** [[Related Paper](https://arxiv.org/abs/2102.02915)]

*ASSUMTION 2* : By **matching the time step** of the simulation and the actual robot **equally** at 0.05 seconds, it can be assumed that the Sim-to-Real gap is caused **only by the inaccuracy of motor modeling.**

![image](https://user-images.githubusercontent.com/74540268/170244886-0cfbc468-01b6-4249-bf97-935bc9a298a0.png)

**Setting** : With the **main body** of the robot is **fixed**, the **same action sequence** is applied to the simulation and the actual robot.


## 2. Training

**RL algorithm** : PPO

**State** : Current Joint position(18) + Action history(18*3) + Command(3)

**Command** : One hot vector (100 : go-straight , 010 : turn left , 001 : turn right)

**Action** : Joint position(18)
* Joint position is continuous (−150 ≤ 𝜃 ≤ 150) degree

## 3. Results - (Motor parameter tuning)
![image](https://user-images.githubusercontent.com/74540268/170244806-2c1a8094-6b53-4e61-9eea-16f15b84b2a5.png)



## 4. Results - (Simulation)
![rotate](https://user-images.githubusercontent.com/74540268/170244381-a976e5b8-544c-467a-804a-087c82f52eb6.gif) ![walking](https://user-images.githubusercontent.com/74540268/170244255-4d7dc8e4-c94e-49ee-8e1e-5bdd66be27f4.gif)


## 5. Results - (Real robot deployment)

**Video : [YOUTUBE link](https://www.youtube.com/watch?v=ApI5J0-24kw)**

You can watch **video** through the link above!


