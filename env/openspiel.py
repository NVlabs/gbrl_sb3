##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from shimmy import OpenSpielCompatibilityV0
except:
    OpenSpielCompatibilityV0 = None

LIARS_DICE_BIDS = {0: 'No bid',
                   1: '1 die shows 1',
                   2: '1 die shows 2',
                   3: '1 die shows 3',
                   4: '1 die shows 4',
                   5: '1 die shows 5',
                   6: '1 die shows 6',
                   7: '2 die shows 1',
                   8: '2 die shows 2',
                   9: '2 die shows 3',
                   10: '2 die shows 4',
                   11: '2 die shows 5',
                   12: '2 die shows 6'
                   }

def liars_dice_wrapper(obs: np.ndarray, player: str):
    player_identity = player
    dice_obs = obs[2:8]
    bid_obs = obs[8:20]
    dice = np.argmax(dice_obs) + 1
    bid_history =  np.argmax(bid_obs) + 1 if bid_obs.sum() > 0 else 0
    bid_history = LIARS_DICE_BIDS[bid_history]
    liar = 'Liar Called' if bool(obs[-1]) else 'Liar not Called'
    return np.array([player_identity, dice, bid_history, liar], dtype=object)

def kuhn_poker_wrapper(obs: np.ndarray, player: str):
    card_obs = obs[2:5]
    cards = {0: 'Jack', 1: 'Queen', 2: 'King'}
    card = cards[np.argmax(card_obs)]
    if player == 'player_0':
        assert obs[0] == 1
        info = [card, obs[5], obs[6]]
    else:
        info = [card, obs[6], obs[5]]
    
    return np.array([info], dtype=object).flatten()


def blackjack_wrapper(obs: np.ndarray, player: str):
    terminal = bool(obs[2])
    dealer_aces = obs[3:8]
    player_aces = obs[8:13]
    n_aces_player = np.argmax(player_aces)
    n_aces_dealer = np.argmax(dealer_aces)
    player_enc = np.zeros(4)
    dealer_cards = obs[13:65]
    dealer_enc = np.zeros(4)
    player_cards = obs[65:]

    for i in range(4):
        if i < 3:
            player_set_cards = player_cards[i*13:(i + 1)*13]
            dealer_set_cards = dealer_cards[i*13:(i + 1)*13]
        else:
            player_set_cards = player_cards[i*13:]
            dealer_set_cards = dealer_cards[i*13:]
        player_set_cards[player_set_cards > 10] = 10
        dealer_set_cards[dealer_set_cards > 10] = 10
        player_enc[i] = np.sum(player_set_cards[:-1])
        dealer_enc[i] = np.sum(dealer_set_cards[:-1])
    # dealer
    if player == 'player_0':
        info = np.array([str(terminal), n_aces_dealer, n_aces_player, np.sum(dealer_enc), np.sum(player_enc)], dtype=object)
    # player
    else:
        info = np.array([str(terminal), n_aces_player, n_aces_dealer, np.sum(player_enc), np.sum(dealer_enc)], dtype=object)
    return info

def connect_four_wrapper(obs: np.ndarray, player: str):
    new_obs = np.empty(6*7, dtype=object)
    new_obs[(obs[2] == 1).flatten()] = 'empty'
    new_obs[(obs[0] == 1).flatten()] = 'player' if player == 'player_0' else 'opponent'
    new_obs[(obs[1] == 1).flatten()] = 'opponent' if player == 'player_0' else 'player'
    return new_obs

def hanabi_wrapper(obs: np.ndarray, player: str):
    colors = {0: 'Red' , 1: 'Yellow', 2: 'Green', 3: 'White', 4: 'Blue'}
    offset = 0
    n_cards = 0
    player_hints = np.empty(6, dtype=object)
    player_hints[:] = 'Unknown'
    for card in range(5):
        card_one_hot = obs[offset:offset+25]
        card_idx = np.where(card_one_hot > 0)[0]
        if len(card_idx) > 0:
            color = card_idx[0] // 5
            rank = card_idx[0] % 5
            player_hints[card] = f'{colors[color]} {rank + 1}'
            n_cards += 1
        offset += 25
    player_hints[-1] = n_cards
    offset += 2
    observer_missing_card = obs[125]
    deck_size = np.sum(obs[offset:offset+40])
    offset += 40
    fireworks = np.empty(5, dtype=object)
    fireworks[:] = 'Unknown'
    for color_idx in range(5):
        firework_obs = obs[offset:offset+5]
        firework_idx = np.where(firework_obs > 0)[0]
        if len(firework_idx) > 0:
            fireworks[color_idx] = f'{colors[color]} {firework_idx + 1}'

        offset += 5
    n_information_tokens = np.sum(obs[offset:offset+8])
    offset += 8 
    n_life_tokens = np.sum(obs[offset:offset+3])
    offset += 3
    discarded = np.zeros(25, dtype=object)
    cards_per_rank = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}
    for color_idx in range(5):
        for rank in range(5):
            discarded_cards = obs[offset:offset+cards_per_rank[rank]]
            discarded[color_idx*5 + rank] = np.sum(discarded_cards)
            offset += cards_per_rank[rank]
    acting_player = np.where(obs[offset:offset+2] > 0)[0]
    if len(acting_player) > 0:
        acting_player = 'me' if acting_player[0] == player else 'other player'
    else:
        acting_player = 'None'
    offset += 2
    plays = {0: 'play', 1: 'discard', 2: 'reveal color', 3: 'reveal rank'}
    last_action = np.where(obs[offset:offset+4] > 0)[0]
    if len(last_action) > 0:
        last_action = plays[last_action[0]]
    else:
        last_action = 'None'
    offset += 4
    target_player = np.where(obs[offset:offset+2] > 0)[0]
    if len(target_player) > 0:
        target_player = 'me' if target_player[0] == 0 else 'other player'
    else:
        target_player = 'None'
    offset += 2
    color_reveal = np.where(obs[offset:offset+5] > 0)[0]
    if len(color_reveal) > 0:
        color_reveal = colors[color_reveal[0]]
    else:
        color_reveal = 'None'
    offset += 5
    rank_reveal = np.where(obs[offset:offset+5] > 0)[0]
    if len(rank_reveal) > 0:
        rank_reveal = rank_reveal[0] + 1
    else:
        rank_reveal = 'None'
    offset += 5
    outcome_reveal = np.empty(5, dtype=object)
    outcome_reveal[:] = 'None'
    outcome_reveals = np.where(obs[offset:offset+5] > 0)[0]
    if len(outcome_reveals) > 0:
        for i, revealed_idx in enumerate(outcome_reveals):
            outcome_reveal[i] = 'Card  ' + str(revealed_idx + 1)
    offset += 5
    position_reveal = np.where(obs[offset:offset+5] > 0)[0]
    if len(position_reveal) > 0:
        position_reveal = 'Card  ' + str(position_reveal[0] + 1)
    else:
        position_reveal = 'None'
    offset += 5
    card_reveal = np.where(obs[offset:offset+25] > 0)[0]
    if len(card_reveal) > 0:
        color = card_reveal[0] // 5
        rank = card_reveal[0] % 5
        card_reveal = f'{colors[color]} {rank + 1}'
    else:
        card_reveal = 'None'
    offset += 25
    action_success = np.where(obs[offset:offset+2] > 0)[0]
    action_information = {0: 'Score', 1: 'Information token'}
    if len(action_success) > 0:
        action_success = action_information[action_success[0]]
    else:
        action_success = 'Fail'
    offset += 2

    info = np.concatenate([player_hints, np.array([str(bool(observer_missing_card)), deck_size]), fireworks, 
                           np.array([n_information_tokens, n_life_tokens]), discarded, 
                           np.array([acting_player, last_action, target_player, color_reveal, rank_reveal, position_reveal, card_reveal, action_success]),
                           outcome_reveal])
    # encoding common knowledge
    for player_id in range(2):
     for cards in range(5):
        plausible_cards = obs[offset:offset+25]
        info = np.append(info, plausible_cards)
        offset += 25
        card_color = np.where(obs[offset:offset+5] > 0)[0]
        if len(card_color) > 0:
            card_color = colors[card_color[0]]
        else:
            card_color = 'None'
        offset += 5
        card_rank = np.where(obs[offset:offset+5] > 0)[0]
        if len(card_rank) > 0:
            card_rank = f'Rank {card_rank[0] + 1}'
        else:
            card_rank = 'None'
        offset += 5
        if f'player_{player_id}' == player:
            card_color += ' (me)'
            card_rank += ' (me)'
        else:
            card_color += ' (other player)'
            card_rank += ' (other player)'
        info = np.append(info, np.array([card_color, card_rank], dtype=object))
    return info


MIXED_SIZES = {'liars_dice': 4, 'kuhn_poker': 3, 'blackjack': 5, 'connect_four': 42, 'hanabi': 323}
OBS_WRAPPERS = {'liars_dice': liars_dice_wrapper, 'kuhn_poker': kuhn_poker_wrapper,
                'blackjack': blackjack_wrapper, 'connect_four': connect_four_wrapper, 
                'hanabi': hanabi_wrapper}

# tiny_bridge_2p
# hanabi
# matrix_pd
# blackjack
# connect_four


class OpenSpielGymEnv(gym.Env):
    def __init__(self, env_id: str, is_mixed:bool = True, **env_kwargs):

        self.env = OpenSpielCompatibilityV0(game_name=env_id, **env_kwargs)  # type: ignore[arg-type]
        self.observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.action_space = self.env.action_space(self.env.possible_agents[0])
        self.num_envs = 1
        self.env_id = env_id
        self.agent_selection = None
        # is_mixed = False
        self.is_mixed = is_mixed
        if is_mixed:
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(MIXED_SIZES.get(env_id, 0), ), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed, options)
        obs = self.observe(self.env.agent_selection)
        if self.is_mixed:
            obs = OBS_WRAPPERS[self.env_id](obs, self.env.agent_selection)
        self.agent_selection = self.env.agent_selection
        return obs, {'player': self.env.agent_selection}
    
    def seed(self, seed):
        pass
    
    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return self.env.observe(agent)

    def render(self):
        return self.env.render()
    
    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, done, info."""
        self.env.step(action)
        new_obs, reward, terminated, truncated, info = self.env.last()
        if self.is_mixed:
            new_obs = OBS_WRAPPERS[self.env_id](new_obs, self.env.agent_selection)
        self.agent_selection = self.env.agent_selection
        if self.env.game_name == 'hanabi':
            reward = np.sum([v for k, v in self.env._cumulative_rewards.items()])
        return new_obs, reward, terminated, truncated, info
    
    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return self.env.infos[self.env.agent_selection]["action_mask"]
    
    def close(self):
        return self.env.close()
    

class ConnectFourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False):
        super(ConnectFourCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        # Additional layers can follow as needed
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

   