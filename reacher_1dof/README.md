## Experiment 2: Towards Energy Awareness
In this experiment the benefits of including informationn related to the virtual energy tank in the RL framework's state space and reward function are highlighted. The information is used by the RL algorithm to come-up with a policy which answers the question of [affordibility](https://www.frontiersin.org/articles/10.3389/fnint.2018.00006/full) in terms of energy requirements. In general, an action is deemed affordable if the robot recieves "enough" energy while executing it. How much energy is "enough" is something which is determined while training.  

A 1-DOF robot's task is to reach a target located at the other end of the arena. It can do so by either rotation clockwise or anticlockwise. Both paths leading to the target are obstructed by movable obstacles with different masses and the robot is not equipped with any type of force or pressure sensor. The controller of the robot which is represented by an RL agent needs to decide whether it is affordable to displace an obstacle with a certain mass in-order to reach the target, based on the change of the energy level in energy tank. If it is affordable then the controller can continue towards the target by moving the obstacle, else the controller needs to change its course of action and try reaching the target from the other side. Thus, the controller needs to sense the level of energy in the energy tank in order to sense the presence of the obstacle, estimate its mass and take actions accordingly.  

The affordibility of an action can be altered by changing the energy constant used to refill the energy tank 


![](imgs/reacher_1dof.gif)

## Setup
A custom design and simple environment was used in-order to highlight and emphasize the "energy-aware" aspect of the framework and ignore the other affects.  

### Energy tank
In beginning of each trial the tank is initialized with 5J of energy. The maximum limit is set to 8J. Since, in this setup the robot contains just 1 actuator, the energy leaving the tank corresponding to the actuator is equal to the total energy leaving the tank.  

### RL algorithm
The state space and action space for RL algorithm is similar to experiment 1. The state space also contains the joint information from the previous time step.  

More details can be viewed in the section 5 of the [thesis](http://essay.utwente.nl/88729/1/Chaturvedi_MA_EEMCs.pdf).

## How to run
After cloning and installing the dependencies, run the main.py python file to start learning
```bash
python3 main.py
```
After training is complete the learned policy can be tested using te infer.py program as:
```bash
python3 infer.py
```
