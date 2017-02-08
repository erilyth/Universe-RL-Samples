#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

Deep Q-Learning - Reinforcement Learning

"""

import gym
import universe

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd

load_model = 1
backup_iter = 500
save_iter = 5
memory_clear = 100

# Q-Model
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2))
model.compile(loss='mse', optimizer=sgd(lr=0.0001))

if load_model == 1:
	model.load_weights("cartpole-v0.keras")

env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1', force=True)
replay_memory = []
gamma = 0.9 # Future reward decrement
epsilon = 0.2 # Probability of selecting random action
epsilon_min = 0.0 # Minimum random action selection probability
episodes = 500
epsilon_decay = (epsilon - epsilon_min) / episodes # Random action selection probability decay

for episode in range(episodes):
	observation = env.reset()
	observation = np.reshape(observation, [1, 4])
	for time_t in range(5000):
		print epsilon
		env.render()
		# Action space is either 0 or 1 for cartpole
		# print env.action_space
		action = model.predict(observation)
		print action
		action = np.argmax(action[0])
		if np.random.uniform(0,1) < epsilon:
			# Either 0 or 1 sample the action randomly
			action = np.random.randint(2)
		#action = env.action_space.sample()
		# print "Action:", action
		observation_old = observation
		observation, reward, done, info = env.step(action)
		observation = np.reshape(observation, [1, 4])
		print observation
		replay_memory.append([observation_old, action, reward, observation])
		if done:
			print 'Episode finished'
			break
	print "Replay Memory Size:", len(replay_memory)
	indices = np.random.choice(len(replay_memory), min(500, len(replay_memory)))
	for mem_idx in indices:
		mem = replay_memory[mem_idx]
		observation_old = mem[0]
		action = mem[1]
		reward = mem[2]
		observation = mem[3]
		target = reward
		if mem_idx != len(replay_memory) - 1: 
			target = reward + gamma * np.amax(model.predict(observation)[0])
		print "Target:", target
		target_f = model.predict(observation_old)
		target_f[0][action] = target
		model.fit(observation_old, target_f, nb_epoch=1, verbose=0)
	if episode % save_iter == 0:
		model.save_weights("cartpole-v0.keras")
	if episode % backup_iter == 0:
		model.save_weights("cartpole_backup" + str(episode) + "-v0.keras")
	if episode % memory_clear == 0:
		replay_memory = []
	epsilon -= epsilon_decay

env.monitor.close()
# Upload onto gym
last_chars = raw_input("Enter last two characters of key: ")
gym.scoreboard.api_key = 'sk_Ai0CaXYKRRS4XX5mCdlJ' + last_chars
gym.upload('/tmp/cartpole-experiment-1')