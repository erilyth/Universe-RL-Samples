#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

Reinforcement Learning using best action based policy updates

"""

import gym
from gym import wrappers
import universe

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import sgd, Adam, RMSprop

load_model = 1
backup_iter = 499
save_iter = 5
memory_clear = 10000

# Reinforcement Learning - Deep-Q learning
model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Activation('tanh'))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
model.compile(loss='mse', optimizer=optimizer)

if load_model == 1:
	model.load_weights("cartpole-v0.keras")

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

print env.action_space, env.observation_space

replay_memory = []
gamma = 0.995 # Future reward decrement
epsilon = 0.5 # Probability of selecting random action
epsilon_decay = 0.995 # Minimum random action selection probability
episodes = 50 # Actual episodes performed are episodes * batch_size
batch_size = 32
#epsilon_decay = (epsilon - epsilon_min) / episodes # Random action selection probability decay

for episode in range(episodes):
	for batch_idx in range(batch_size):
		observation = env.reset()
		observation = observation.reshape([1, 4])
		total_reward = 0.0
		current_replay_mem = []
		for time_t in range(250):
			print "Epsilon: ", epsilon, " && Episode: ", episode, " && Batch idx: ", batch_idx
			env.render()
			# Action space is either 0 or 1 for cartpole
			#print env.action_space
			action = model.predict(observation)
			print "Action predicted:", action
			action = np.argmax(action[0])
			if np.random.uniform(0,1) < epsilon:
				# Either 0 or 1 sample the action randomly
				action = np.random.randint(2)
			#action = env.action_space.sample()
			# print "Action:", action
			observation_old = observation
			observation, reward, done, info = env.step(action)
			total_reward += reward
			observation = observation.reshape([1, 4])
			print "Observation:", observation
			print "Reward:", reward
			# print "Observation:", observation
			current_replay_mem.append([observation_old, action, reward, observation])
			if done:
				print 'Episode finished'
				break
		if epsilon >= 0.01:
			epsilon *= epsilon_decay
		for mem_id in range(len(current_replay_mem)):
			mem_vals = current_replay_mem[mem_id]
			mem_vals[2] = total_reward
			replay_memory.append(mem_vals)
	#print "Replay Memory Size:", len(replay_memory)
	total_rewards = []
	for mem_idx in range(len(replay_memory)):
		total_rewards.append(replay_memory[mem_idx][2])
	threshold = np.percentile(total_rewards, 50)
	for mem_idx in range(len(replay_memory)):
		mem = replay_memory[mem_idx]
		observation_old = mem[0]
		action = mem[1]
		action_onehot = np.zeros((1,2))
		action_onehot[0][action] = 1
		reward = mem[2]
		observation = mem[3]
		if reward > threshold:
			model.fit(observation_old, action_onehot, nb_epoch=1, verbose=0)
	replay_memory = []
	if episode % save_iter == 0:
		model.save_weights("cartpole-v0.keras")
	if episode % backup_iter == 0:
		model.save_weights("cartpole_backup" + str(episode) + "-v0.keras")
	if episode % memory_clear == 0:
		replay_memory = []

env.monitor.close()
# Upload onto gym
last_chars = raw_input("Enter last two characters of key: ")
gym.scoreboard.api_key = 'sk_Ai0CaXYKRRS4XX5mCdlJ' + last_chars
gym.upload('/tmp/cartpole-experiment-1')
