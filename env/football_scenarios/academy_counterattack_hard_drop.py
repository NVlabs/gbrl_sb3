# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gfootball.scenarios import *
import numpy as np


def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetBallPosition(0.26, -0.11)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.672, -0.19576, e_PlayerRole_LB)
  builder.AddPlayer(-0.75, -0.06356, e_PlayerRole_CB)
  builder.AddPlayer(-0.75, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.672, 0.19576, e_PlayerRole_RB)
  builder.AddPlayer(-0.434, -0.10568, e_PlayerRole_CM)
  builder.AddPlayer(-0.434, 0.10568, e_PlayerRole_CM)
  builder.AddPlayer(0.5, -0.3161, e_PlayerRole_CM)
  builder.AddPlayer(0.25, -0.1, e_PlayerRole_LM)
  builder.AddPlayer(0.25, 0.1, e_PlayerRole_RM)
  builder.AddPlayer(0.35, 0.316102, e_PlayerRole_CF)

  random_drop = np.random.randint(0, high=10)
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  if random_drop != 0:
    builder.AddPlayer(0.128, -0.19576, e_PlayerRole_LB)
  if random_drop != 1:
    builder.AddPlayer(-0.4, -0.06356, e_PlayerRole_CB)
  if random_drop != 2:
    builder.AddPlayer(-0.4, 0.063559, e_PlayerRole_CB)
  if random_drop != 3:
    builder.AddPlayer(0.128, -0.19576, e_PlayerRole_RB)
  if random_drop != 4:
    builder.AddPlayer(0.365, -0.10568, e_PlayerRole_CM)
  if random_drop != 5:
    builder.AddPlayer(0.282, 0.0, e_PlayerRole_CM)
  if random_drop != 6:
    builder.AddPlayer(0.365, 0.10568, e_PlayerRole_CM)
  if random_drop != 7:
    builder.AddPlayer(0.54, -0.3161, e_PlayerRole_LM)
  if random_drop != 8:
    builder.AddPlayer(0.51, 0.0, e_PlayerRole_RM)
  if random_drop != 9:
    builder.AddPlayer(0.54, 0.316102, e_PlayerRole_CF)