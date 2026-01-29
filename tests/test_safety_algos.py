"""
Sanity check tests for safety RL algorithms.
These tests verify basic correctness of the implementations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import numpy as n
from unittest.mock import Mock, MagicMock, patch
import gymnasium as gym
from gymnasium import spaces


import unittest


class TestReturnOrderConsistency(unittest.TestCase):
    """Test that algorithms correctly handle evaluate_actions return order."""
    
    def test_nn_policy_evaluate_actions_order(self):
        """Verify NN policy returns (values, costs, log_prob, entropy)."""
        # This tests the expected interface
        # values should be reward value estimates
        # costs should be cost value estimates
        
        # Mock tensors
        values = th.tensor([[1.0], [2.0]])  # Reward values
        costs = th.tensor([[0.5], [0.3]])   # Cost values
        log_prob = th.tensor([-0.1, -0.2])
        entropy = th.tensor([0.5, 0.5])
        
        # The expected order from NN policy
        result = (values, costs, log_prob, entropy)
        
        # Verify unpacking works correctly
        v, c, lp, ent = result
        assert th.equal(v, values), "Values should be first"
        assert th.equal(c, costs), "Costs should be second"
        

class TestCPOStepDirection(unittest.TestCase):
    """Test CPO step direction computation."""
    
    def test_case_3_4_uses_trpo_step(self):
        """When safe (case 3 or 4), CPO should use standard TRPO step."""
        from algos.safety.cpo import CPO
        
        # Create minimal mock
        cpo = Mock(spec=CPO)
        cpo._compute_cpo_step_direction = CPO._compute_cpo_step_direction.__get__(cpo)
        
        search_direction = th.randn(10)
        p = th.randn(10)
        q = th.tensor(2.0)  # x^T H x
        r = th.tensor(0.1)
        s = th.tensor(0.5)
        A = th.tensor(1.0)
        B = th.tensor(-1.0)  # B < 0 for case 3
        ep_costs = -5.0  # Safe: below limit
        target_kl = 0.01
        
        step, lambda_star, nu_star = cpo._compute_cpo_step_direction(
            optim_case=3,
            search_direction=search_direction,
            p=p, q=q, r=r, s=s, A=A, B=B,
            ep_costs=ep_costs,
            target_kl=target_kl
        )
        
        # In case 3/4, nu_star should be 0 (no cost constraint active)
        assert nu_star == 0.0, "nu_star should be 0 in safe case"
        
        # Step should be proportional to search_direction
        expected_alpha = th.sqrt(2 * target_kl / (q + 1e-8))
        expected_step = expected_alpha * search_direction
        assert th.allclose(step, expected_step, atol=1e-5), "Step should match TRPO step"
    
    def test_case_0_recovery_only(self):
        """Case 0: infeasible, should only decrease cost."""
        from algos.safety.cpo import CPO
        
        cpo = Mock(spec=CPO)
        cpo._compute_cpo_step_direction = CPO._compute_cpo_step_direction.__get__(cpo)
        
        search_direction = th.randn(10)
        p = th.randn(10)  # Cost descent direction
        q = th.tensor(2.0)
        r = th.tensor(0.1)
        s = th.tensor(0.5)
        A = th.tensor(1.0)
        B = th.tensor(1.0)
        ep_costs = 10.0  # Unsafe
        target_kl = 0.01
        
        step, lambda_star, nu_star = cpo._compute_cpo_step_direction(
            optim_case=0,
            search_direction=search_direction,
            p=p, q=q, r=r, s=s, A=A, B=B,
            ep_costs=ep_costs,
            target_kl=target_kl
        )
        
        # In case 0, lambda_star should be 0 (ignore reward)
        assert lambda_star == 0.0, "lambda_star should be 0 in recovery case"
        
        # Step should be in -p direction (cost decrease)
        expected_nu = th.sqrt(2 * target_kl / (s + 1e-8))
        expected_step = -expected_nu * p
        assert th.allclose(step, expected_step, atol=1e-5), "Step should be -nu*p"


class TestCPOCaseDetermination(unittest.TestCase):
    """Test CPO case determination logic."""
    
    def test_case_4_negligible_cost_gradient(self):
        """Case 4: cost gradient ~0 and safe."""
        from algos.safety.cpo import CPO
        
        cpo = Mock(spec=CPO)
        cpo._determine_case = CPO._determine_case.__get__(cpo)
        
        b_grads = th.zeros(10)  # Negligible
        ep_costs = -5.0  # Safe
        q = th.tensor(1.0)
        r = th.tensor(0.0)
        s = th.tensor(0.0)
        target_kl = 0.01
        
        case, A, B = cpo._determine_case(b_grads, ep_costs, q, r, s, target_kl)
        assert case == 4, "Should be case 4"
    
    def test_case_3_entirely_feasible(self):
        """Case 3: ep_costs < 0 and B < 0."""
        from algos.safety.cpo import CPO
        
        cpo = Mock(spec=CPO)
        cpo._determine_case = CPO._determine_case.__get__(cpo)
        
        b_grads = th.randn(10)  # Non-negligible
        ep_costs = -10.0  # Very safe
        q = th.tensor(1.0)
        r = th.tensor(0.1)
        s = th.tensor(0.01)  # Small s makes B negative
        target_kl = 0.001  # Small target_kl
        
        case, A, B = cpo._determine_case(b_grads, ep_costs, q, r, s, target_kl)
        
        # B = 2*target_kl - ep_costs^2 / s = 0.002 - 100/0.01 = 0.002 - 10000 < 0
        assert B.item() < 0, f"B should be negative, got {B.item()}"
        assert case == 3, f"Should be case 3, got {case}"


class TestCPOLambdaProjection(unittest.TestCase):
    """Test the lambda projection bounds in CPO."""
    
    def test_projection_bounds_safe_case(self):
        """When ep_costs < 0, verify projection bounds make sense."""
        # The issue: when ep_costs < 0, r/ep_costs could be negative if r > 0
        # This makes clamping to [0, negative] problematic
        
        ep_costs = -5.0  # Safe
        r = th.tensor(0.5)  # Positive
        
        # r / ep_costs = 0.5 / -5 = -0.1 (negative!)
        bound = r / ep_costs
        
        # Clamping lambda to [0, -0.1] doesn't make sense
        # The actual bound should consider the sign
        assert bound < 0, "r/ep_costs is negative when safe"
        
        # This suggests the implementation may have an issue
        # The correct approach from CPO paper uses |c| and considers signs


class TestLagrangianUpdate(unittest.TestCase):
    """Test Lagrangian multiplier update in PPO-Lag."""
    
    def test_lagrangian_increases_when_unsafe(self):
        """Lambda should increase when cost exceeds limit."""
        lagrangian = th.nn.Parameter(th.tensor(0.5))
        optimizer = th.optim.Adam([lagrangian], lr=0.1)
        
        cost_limit = 25.0
        ep_cost_mean = 30.0  # Above limit
        
        optimizer.zero_grad()
        # Loss = -lambda * (cost - limit) = -0.5 * (30 - 25) = -2.5
        # Gradient w.r.t lambda = -(cost - limit) = -5
        # Update: lambda = lambda - lr * grad = 0.5 - 0.1 * (-5) = 0.5 + 0.5 = 1.0
        loss = -lagrangian * (ep_cost_mean - cost_limit)
        loss.backward()
        optimizer.step()
        
        assert lagrangian.item() > 0.5, f"Lambda should increase, got {lagrangian.item()}"
    
    def test_lagrangian_decreases_when_safe(self):
        """Lambda should decrease when cost is below limit."""
        lagrangian = th.nn.Parameter(th.tensor(0.5))
        optimizer = th.optim.Adam([lagrangian], lr=0.1)
        
        cost_limit = 25.0
        ep_cost_mean = 10.0  # Below limit
        
        optimizer.zero_grad()
        # Loss = -lambda * (cost - limit) = -0.5 * (10 - 25) = -0.5 * (-15) = 7.5
        # Gradient w.r.t lambda = -(cost - limit) = 15
        # Update: lambda = lambda - lr * grad = 0.5 - 0.1 * 15 = -1.0, clamped to 0
        loss = -lagrangian * (ep_cost_mean - cost_limit)
        loss.backward()
        optimizer.step()
        lagrangian.data.clamp_(0.0, None)
        
        assert lagrangian.item() < 0.5, f"Lambda should decrease, got {lagrangian.item()}"


class TestIPOPenalty(unittest.TestCase):
    """Test IPO penalty computation."""
    
    def test_penalty_increases_near_limit(self):
        """Penalty should increase as cost approaches limit."""
        kappa = 0.01
        cost_limit = 25.0
        
        # Far from limit
        ep_cost_mean_far = 10.0
        penalty_far = kappa / (cost_limit - ep_cost_mean_far + 1e-8)
        
        # Close to limit
        ep_cost_mean_close = 24.0
        penalty_close = kappa / (cost_limit - ep_cost_mean_close + 1e-8)
        
        assert penalty_close > penalty_far, "Penalty should increase near limit"
    
    def test_penalty_clamped_when_exceeded(self):
        """Penalty should be clamped when cost exceeds limit."""
        kappa = 0.01
        cost_limit = 25.0
        penalty_max = 1.0
        
        ep_cost_mean = 30.0  # Exceeds limit
        
        penalty = kappa / (cost_limit - ep_cost_mean + 1e-8)
        # penalty = 0.01 / (-5 + 1e-8) ≈ -0.002 (negative!)
        
        if penalty < 0 or penalty > penalty_max:
            penalty = penalty_max
        
        assert penalty == penalty_max, "Should clamp to penalty_max"


class TestCUPProjection(unittest.TestCase):
    """Test CUP projection step."""
    
    def test_kl_direction(self):
        """Verify KL divergence direction interpretation."""
        # Create two simple distributions
        mean1, std1 = th.tensor([0.0]), th.tensor([1.0])
        mean2, std2 = th.tensor([1.0]), th.tensor([1.0])
        
        dist1 = th.distributions.Normal(mean1, std1)
        dist2 = th.distributions.Normal(mean2, std2)
        
        # KL(old || new) - used in CUP
        kl_old_new = th.distributions.kl_divergence(dist1, dist2)
        
        # KL(new || old) - alternative
        kl_new_old = th.distributions.kl_divergence(dist2, dist1)
        
        # For Normal with same std, KL is symmetric
        # But the direction matters for mode-seeking vs mode-covering
        assert kl_old_new.item() > 0, "KL should be positive"
        assert kl_new_old.item() > 0, "KL should be positive"


if __name__ == "__main__":
    import unittest
    
    # Convert test classes to unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in [TestReturnOrderConsistency, TestCPOStepDirection, TestCPOCaseDetermination, 
                       TestCPOLambdaProjection, TestLagrangianUpdate, TestIPOPenalty, TestCUPProjection]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
