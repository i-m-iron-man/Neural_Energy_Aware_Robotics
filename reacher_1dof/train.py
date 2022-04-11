from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model

BATCH_SIZE = 512
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

	def __init__(self, state_dim, action_dim, action_lim, ram):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.gpu = torch.cuda.is_available()
		print("gpu:",self.gpu)
		if self.gpu:
			self.cuda()



		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE/2)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		
	def get_exploitation_action(self, state):
		if self.gpu:
			state = Variable(torch.from_numpy(state)).cuda()
			action = self.target_actor.forward(state).cpu().detach()
		else:
			state = Variable(torch.from_numpy(state))
			action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		if self.gpu:
			state = Variable(torch.from_numpy(state)).cuda()
			action = self.actor.forward(state).cpu().detach()
		else:
			state = Variable(torch.from_numpy(state))
			action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		s1,a1,r1,s2,done = self.ram.sample(BATCH_SIZE)
		
		if self.gpu:
			s1 = Variable(torch.from_numpy(s1)).cuda()
			a1 = Variable(torch.from_numpy(a1)).cuda()
			r1 = Variable(torch.from_numpy(r1)).cuda()
			s2 = Variable(torch.from_numpy(s2)).cuda()
			done = Variable(torch.from_numpy(done)).cuda()
		else:
			s1 = Variable(torch.from_numpy(s1))
			a1 = Variable(torch.from_numpy(a1))
			r1 = Variable(torch.from_numpy(r1))
			s2 = Variable(torch.from_numpy(s2))
			done = Variable(torch.from_numpy(done))


		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		y_expected = r1 + GAMMA*next_val*(1-done)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)

		return loss_critic.item(),loss_actor.data.item()

	def save_models(self, episode_count):
		torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
		print('Models saved successfully')

	def load_models(self, episode):
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print('Models loaded succesfully')

	def cuda(self):
		self.actor.cuda()
		self.target_actor.cuda()
		self.critic.cuda()
		self.target_critic.cuda()

	def reset(self):
		self.noise.reset()