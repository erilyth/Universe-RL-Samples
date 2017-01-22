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
from gym import wrappers
import universe

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import sgd, Adam

load_model = 1
backup_iter = 499
save_iter = 5
memory_clear = 1000

# Reinforcement Learning - Deep-Q learning
model = Sequential()
model.add(Dense(128, input_dim=8))
model.add(Activation('tanh'))
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(4))
model.compile(loss='mse', optimizer=Adam(lr=0.0002))

if load_model == 1:
	model.load_weights("lunarlander-v2.keras")

env = gym.make('LunarLander-v2')
env = wrappers.Monitor(env, '/tmp/lunarlander-experiment-1', force=True)

print env.action_space.n, env.observation_space.shape

replay_memory = []
gamma = 0.995 # Future reward decrement
epsilon = 0.2 # Probability of selecting random action
epsilon_min = 0.1 # Minimum random action selection probability
episodes = 200
state = 0
epsilon_decay = (epsilon - epsilon_min) / episodes # Random action selection probability decay

for episode in range(episodes):
	state = 0
	observation = env.reset()
	observation = np.reshape(observation, [1, 8])
	for time_t in range(1000):
		print "Epsilon:", epsilon
		env.render()
		# Action space is either 0 or 1 for cartpole
		#print env.action_space
		action = model.predict(observation)
		print "Action predicted:", action
		action = np.argmax(action[0])
		if np.random.uniform(0,1) < epsilon:
			# Either 0 or 1 sample the action randomly
			action = np.random.randint(4)
		#action = env.action_space.sample()
		# print "Action:", action
		observation_old = observation
		observation, reward, done, info = env.step(action)
		print "Reward:", reward
		observation = np.reshape(observation, [1, 8])
		if observation[0][6] == 1.0 and observation[0][7] == 1.0:
			# If the lander has already landed, don't let it waste time there. The game does not end by deafult - Bug?
			if state == 0:
				state = 1
			elif state == 1:
				replay_memory.append([observation_old, action, reward+30, observation])
				print 'Episode finished early?!'
				break
		# print "Observation:", observation
		replay_memory.append([observation_old, action, reward, observation])
		if done:
			print 'Episode finished'
			break
	#print "Replay Memory Size:", len(replay_memory)
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
		model.save_weights("lunarlander-v2.keras")
	if episode % backup_iter == 0:
		model.save_weights("lunarlander_backup" + str(episode) + "-v2.keras")
	if episode % memory_clear == 0:
		replay_memory = []
	epsilon -= epsilon_decay

env.monitor.close()
# Upload onto gym
last_chars = raw_input("Enter last two characters of key: ")
gym.scoreboard.api_key = 'sk_Ai0CaXYKRRS4XX5mCdlJ' + last_chars
gym.upload('/tmp/lunarlander-experiment-1')
