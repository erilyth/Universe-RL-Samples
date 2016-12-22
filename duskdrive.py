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

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1) # create one flashgames Docker container
observation_n = env.reset()

while True:
  # your agent generates action_n at 60 frames per second
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
