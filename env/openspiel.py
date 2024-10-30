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

try:
    from shimmy import OpenSpielCompatibilityV0
except:
    OpenSpielCompatibilityV0 = None

MIXED_SIZES = {'chess': 134, 'liars_dice': 4, 'kuhn_poker': 3, 'blackjack': 5}


def chess_wrapper(obs: np.ndarray, player: str):
    obs_positions = obs[:13].astype(bool)
    categorical_board = np.empty(8*8, dtype=object)
    i = 0
    if player == 'player_0':
        player_is_black = True 
    else:
        player_is_black = False 
    for pawn_type in ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']:
        black_str = f'black {pawn_type}'
        white_str = f'white {pawn_type}'
        black_str += ' (me)' if player_is_black else ' (opponent)'
        white_str += ' (me)' if not player_is_black else ' (opponent)'
        categorical_board[obs_positions[i].flatten()] = white_str
        categorical_board[obs_positions[i+1].flatten()] = black_str
        i += 2

    categorical_board[obs_positions[i].flatten()] = 'empty'
    repetitions = obs[13].copy().flatten()
    color_to_play = 'white' if (obs[14, 0, 0] == 1) else 'black'
    color_to_play += ' (me)' if player_is_black and (obs[14, 0, 0] == 0) else ' (opponent)'
    n_irreversible_moves = obs[15, 0, 0]
    white_cast_queenside = 'True' if (obs[16, 0, 0] == 1) else 'False' 
    white_cast_kingside = 'True' if (obs[17, 0, 0] == 1) else 'False' 
    
    black_cast_queenside = 'True' if (obs[18, 0, 0] == 1) else 'False' 
    black_cast_kingside = 'True' if (obs[19, 0, 0] == 1) else 'False'

    black_cast_kingside +=  ' (me)' if player_is_black else ' (opponent)'
    black_cast_queenside +=  ' (me)' if player_is_black else ' (opponent)'
    black_cast_kingside +=  ' (me)' if not player_is_black else ' (opponent)'
    white_cast_queenside +=  ' (me)' if not player_is_black else ' (opponent)'
    
    chess_obs = np.concatenate([categorical_board, repetitions, np.array([color_to_play, n_irreversible_moves, 
                                                                          white_cast_queenside, white_cast_kingside,
                                                                          black_cast_queenside, black_cast_kingside])], axis=0, dtype=object)
    return chess_obs

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


OBS_WRAPPERS = {'chess': chess_wrapper, 'liars_dice': liars_dice_wrapper, 'kuhn_poker': kuhn_poker_wrapper,
                'blackjack': blackjack_wrapper}
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
        return new_obs, reward, terminated, truncated, info
    
    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return self.env.infos[self.env.agent_selection]["action_mask"]
    
    def close(self):
        return self.env.close()
    

