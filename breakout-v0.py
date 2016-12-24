#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import gym
import universe

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import sgd, Adam

load_model = 0
backup_iter = 100
save_iter = 5
memory_clear = 100

# Reinforcement Learning - Deep-Q learning
model = Sequential()
model.add(Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', input_shape=(210, 160, 6)))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.2))
model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same'))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same'))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6))
model.compile(loss='mse', optimizer=Adam(lr=0.001))
print model.summary()

if load_model == 1:
	model.load_weights("breakout-v0.keras")

env = gym.make('Breakout-v0')
env.monitor.start('/tmp/breakout-experiment-1', force=True)
replay_memory = []
observation_prev = None
observation_cur = None
gamma = 0.9 # Future reward decrement
epsilon = 0.8 # Probability of selecting random action
epsilon_min = 0.1 # Minimum random action selection probability
episodes = 1000
epsilon_decay = (epsilon - epsilon_min) / episodes # Random action selection probability decay

for episode in range(episodes):
	observation = env.reset()
	observation = observation.reshape([1, 210, 160, 3])
	for time_t in range(100000):
		print "Epsilon:", epsilon
		env.render()
		# Action space is all integers in [0, 5] for breakout
		# print env.action_space
		action = model.predict(np.concatenate([observation, observation], axis=3).copy())
		print "Action predicted:", action
		action = np.argmax(action[0])
		if np.random.uniform(0,1) < epsilon:
			action = np.random.randint(6)
		# action = env.action_space.sample()
		# print "Action:", action
		observation_old = observation
		observation, reward, done, info = env.step(action)
		observation = observation.reshape([1, 210, 160, 3])
		observation_cur = np.concatenate([observation_old, observation], axis=3).copy()
		# print observation
		if observation_prev != None:
			replay_memory.append([observation_prev, action, reward, observation_cur])
		observation_prev = observation_cur.copy()
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
		model.save_weights("breakout-v0.keras")
	if episode % backup_iter == 0:
		model.save_weights("breakout_backup" + str(episode) + "-v0.keras")
	if episode % memory_clear == 0:
		replay_memory = []
	epsilon -= epsilon_decay

env.monitor.close()
# Upload onto gym
last_chars = raw_input("Enter last two characters of key: ")
gym.scoreboard.api_key = 'sk_Ai0CaXYKRRS4XX5mCdlJ' + last_chars
gym.upload('/tmp/breakout-experiment-1')