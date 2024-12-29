##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import random
from gymnasium.envs.registration import register


class BankTransactionEnv(gym.Env):
    """
    Bank Transaction Validation Environment
    - Agent validates sequential transactions (Valid, Invalid, Flag).
    - Transactions have symbolic and numeric features.
    - Rewards depend on correct validation, flagging, and sequence pattern detection.
    - Transactions now include clearer patterns for normal and fraudulent activity.
    - Rules are more nuanced with time, frequency, and account-type constraints.
    - Different rules apply globally and specific rules apply per account type.
    - Mixed observation space supported.
    - Rewards are normalized to ensure total reward is capped at 1.
    """
    def __init__(self, max_transactions=50, n_clients=5, is_mixed=False):
        super(BankTransactionEnv, self).__init__()
        
        self.max_transactions = max_transactions
        self.current_step = 0
        self.is_mixed = is_mixed
        self.n_clients = n_clients
        self.transaction_history = []
        self.client_data = {}
        
        # Symbolic Features
        self.account_types = ['Personal', 'Business', 'VIP']
        self.transaction_types = ['Deposit', 'Withdrawal', 'Transfer']
        self.location_types = ['Local', 'International']
        self.time_of_day = ['Morning', 'Afternoon', 'Night']
        
        # Numeric Features
        self.min_amount = 0
        self.max_amount = 10000
        self.min_balance = 0
        self.max_balance = 50000
        
        # Frequency limits per account type
        self.frequency_limits = {
            'Personal': {'total': 5, 'time': 3},
            'Business': {'total': 15, 'time': 5},
            'VIP': {'total': 10, 'time': 4}
        }
        self.withdrawal_limits = {'Personal': 10000,
                                  'Business': 20000,
                                  'VIP': 25000}
        self.transfer_limits = {'Local':{
                                        'Personal': 10000,
                                        'Business': 20000,
                                        'VIP': 25000},
                                'International':{
                                        'Personal': 5000,
                                        'Business': 10000,
                                        'VIP': 12500},
                                }
    
        # Observation Space
        obs_shape = 14
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_shape,), dtype=np.float32
        )
        # Action Space (Validate, Flag)
        self.action_space = spaces.MultiDiscrete([2, 2])  # Validate (Valid/Invalid), Flag (Yes/No)

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        self.client_data = {}
        self.total_amount = 0
        self.passed_amount = 0
        self.liquidity = 2*self.max_balance
        self.initial_liquidity = 2*self.max_balance
        
        # Initialize clients with random balances and optional history
        for i in range(self.n_clients):
            account_type = random.choice(self.account_types)
            self.client_data[i] = {
                'account_type': account_type,
                'balance': random.uniform(self.min_balance, self.max_balance),
                'freq_total': 0,
                'freq_morning': random.randint(0, 2),
                'freq_afternoon': random.randint(0, 2),
                'freq_night': random.randint(0, 2),
                'withdrawal_sum': random.uniform(0, 5000),
                'flags': 0,
                'blocked': False,
                'since_blocked': 0
            }
            self.client_data[i]['freq_total'] = self.client_data[i]['freq_morning'] + self.client_data[i]['freq_afternoon'] + self.client_data[i]['freq_night']
        return self._get_observation(), {}
    
    def _generate_transaction(self):
        """Generate a single transaction dynamically each step."""
        client_id = random.choice(list(self.client_data.keys()))
        transaction = {
            'client_id': client_id,
            'account_type': self.client_data[client_id]['account_type'],
            'transaction_type': random.choice(self.transaction_types),
            'location': random.choice(self.location_types),
            'time': random.choice(self.time_of_day),
            'amount': random.uniform(self.min_amount, self.max_amount),
            'balance': self.client_data[client_id]['balance'],
        }
        
        # Update frequency
        time_key = f"freq_{transaction['time'].lower()}"
        self.client_data[client_id]['freq_total'] += 1
        self.client_data[client_id][time_key] += 1
        
        return transaction
    
    def _get_observation(self):
        """Construct the observation space."""
        t = self._generate_transaction()
        obs = [
            self.time_of_day.index(t['time']) / len(self.time_of_day),
            t['amount'],
            t['balance'],
            self.client_data[t['client_id']]['freq_total'],
            self.client_data[t['client_id']]['freq_morning'],
            self.client_data[t['client_id']]['freq_afternoon'],
            self.client_data[t['client_id']]['freq_night'],
            self.client_data[t['client_id']]['withdrawal_sum'],
            self.client_data[t['client_id']]['flags'],
            self.liquidity,
        ]
        
        if self.is_mixed:
            obs.extend([
                str(t['account_type']).encode('utf-8'),
                str(t['transaction_type']).encode('utf-8'),
                str(t['location']).encode('utf-8'),
                str(self.client_data[t['client_id']]['blocked']).encode('utf-8')
            ])
        else:
            obs.extend([
                    self.account_types.index(t['account_type']) / len(self.account_types),
                    self.transaction_types.index(t['transaction_type']) / len(self.transaction_types),
                    self.location_types.index(t['location']) / len(self.location_types),
                    int(self.client_data[t['client_id']]['blocked']),
            ])
        self.last_transaction = t
        
        return np.array(obs, dtype=np.float32 if not self.is_mixed else object)
    
    def step(self, action):
        """Take an action: Validate (Valid/Invalid), Flag (Yes/No)."""
        validate_action, flag_action = action
        transaction = self.last_transaction
        client = self.client_data[transaction['client_id']]
        reward = 0
        truncated = False
        self.total_amount += transaction['amount']
        
        # Validation Rules
        if transaction['transaction_type'] == 'Transfer' and transaction['amount'] > self.transfer_limits[transaction['location']][client['account_type']]:
            expected_validation = validate_action == 0
        elif client['freq_total'] > self.frequency_limits[client['account_type']]['total']:
            expected_validation = flag_action == 1
        elif client['freq_night'] > self.frequency_limits[client['account_type']]['time']:
            expected_validation = flag_action == 1
        elif transaction['time'] == 'Night' and transaction['amount'] > 5000:
            expected_validation = flag_action == 1
        elif transaction['transaction_type'] == 'Deposit' and transaction['time'] == 'Morning' and transaction['amount'] > 5000:
            expected_validation = flag_action == 1
        elif self.client_data[transaction['client_id']]['withdrawal_sum'] > 10000 and client['account_type'] == 'Personal':
            expected_validation = flag_action == 1
        elif self.client_data[transaction['client_id']]['withdrawal_sum'] > self.withdrawal_limits[client['account_type']]:
            expected_validation = flag_action == 1
        elif client['flags'] > 0 and transaction['transaction_type'] == 'Withdrawal':
            expected_validation = validate_action == 0
        elif transaction['transaction_type'] == 'Withdrawal' and  transaction['amount'] > self.liquidity:
            expected_validation = validate_action == 0
        elif client['flags'] > 3:
            expected_validation = validate_action == 0
        else:
            expected_validation = validate_action == 1 and flag_action == 0

        if transaction['transaction_type'] != 'Deposit' and (transaction['amount'] > transaction['balance'] or transaction['balance'] <= 0):
            expected_validation = validate_action == 0
        if client['blocked']:
            expected_validation = validate_action == 0

        if validate_action and not client['blocked']:
            self.passed_amount = transaction['amount']
            if transaction['transaction_type'] == 'Deposit':
                self.client_data[transaction['client_id']]['balance'] += transaction['amount']
                self.liquidity += transaction['amount']
            else:
                self.client_data[transaction['client_id']]['balance'] -= transaction['amount']
            
            if transaction['transaction_type'] == 'Withdrawal':
                self.client_data[transaction['client_id']]['withdrawal_sum'] += transaction['amount']
                self.liquidity -= transaction['amount']

            
        # Reward logic
        if expected_validation:
            reward += 1
        else:
            reward -= 1
        
        # if flag_action == 1 and not expected_validation:
        #     reward += 0.5
        # elif flag_action == 1 and expected_validation:
        #     reward -= 0.5
        
        reward = reward / self.max_transactions
        self.client_data[transaction['client_id']]['flags'] = self.client_data[transaction['client_id']]['flags'] + 1 if flag_action == 1 else 0
        self.client_data[transaction['client_id']]['since_blocked'] = self.client_data[transaction['client_id']]['since_blocked'] + 1 if self.client_data[transaction['client_id']]['blocked'] else 0
        # block or reset block
        if validate_action == 1:
            if self.client_data[transaction['client_id']]['since_blocked'] >= 3:
                self.client_data[transaction['client_id']]['blocked'] = False
                self.client_data[transaction['client_id']]['since_blocked'] = 0
        else:
            self.client_data[transaction['client_id']]['blocked'] = True 

        self.current_step += 1
        terminated = self.current_step >= self.max_transactions
        if self.liquidity < 0:
            truncated = True
            reward = -1
        
        # if terminated:
        #     if self.liquidity > 0.5*self.initial_liquidity:
        #         reward += 0.5
        #     else:
        #         reward -= 0.5
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Transaction: {self.transaction_history[-1]}")


def register_bank_validation_tests():
    # PutNear
    # ----------------------------------------

    register(
        id="bank-v0",
        entry_point="env.bank_validation:BankTransactionEnv",
        kwargs={'max_transactions': 50},
    )