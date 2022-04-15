## Experiment 3: linking to Homeostatic Reinforcement Learning
Homeostasis refers to the process of self-regulating the internal states of an organism by it, such that
the internal states are maintained within a certain range. [Homeostatic reinforcement learning(HRL)](https://elifesciences.org/articles/04811) bridges the gap between the associative and homeostatic learning process by proving that seeking reward is equivalent to the fundamental objective of physiological stability. An example of the HRL framework for an organism trying to maintain its glucose and temperature levels in acceptable limits is shown below.  
<img src=imgs/HRL_framework.jpg width="500">  

In the context of this experiment, the energy in the virtual energy tank is considered as the internal homeostatic state and the reward function is defined for maximizing this energy. The change of energy in the energy tank due to environment can be considered as the outcome mentioned in the HRL framework.  
The technique used here also improves training results in case of infinite horizon tasks when the buffer memory of the training hardware is limited, as the training episodes terminate sooner. More details about this aspect can be found in section 6 of the [thesis](http://essay.utwente.nl/88729/1/Chaturvedi_MA_EEMCs.pdf)

![](imgs/cheetah.gif)

### Reward Function

