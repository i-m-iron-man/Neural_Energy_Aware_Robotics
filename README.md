# Neural Energy Aware Robotics Framework

This reposistory contains the code used in my [masters thesis](http://essay.utwente.nl/88729/1/Chaturvedi_MA_EEMCs.pdf), which introduces the idea of dynamic energy budgeting and behavioural shaping in robotics using reinforcement learning. The three sub directories correspond to the three experiment in the thesis.  
The basic idea is to, firstly, use the reward function in reinforcement learning framework to feed the energy tank used in [virtual energy budgeting framework](https://ieeexplore.ieee.org/document/8463174) with positive energy. Secondly, to include components of the energy tank's states such as energy tank level, rate of energy going in, rate of energy going out etc in the state vector used in a reinforcment learning algorithm and reward function, in order to enable complete energy awareneness. This enables maintaining energy level in the energy tank below a saftey threshold throughout the task execution and selective passivity of the robot's controller i.e. the robot's controller is passive with respect to the energy tank when the robot does not aqcuire positive reward from the environment and non-passive otherwise.  
This framework also links the virtual energy methodology to [homeostatic reinforcement leanring](https://elifesciences.org/articles/04811.pdf) by considering energy as a homeostatic regulated state. This is emphasized more in the 'half-cheetah' experiment.

## How to run
First, install dependencies. Use `Python >= 3.8`:
```bash
# clone project   
git clone https://github.com/i-m-iron-man/Neural_Energy_Aware_Robotics.git   

# install dependencies   
cd Neural_Energy_Aware_Robotics/ 
pip install -r requirements.txt
```
Next, go to the respective sub directory to run the experiments


