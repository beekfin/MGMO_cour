import unittest

import numpy as np
import torch

from muzero_continuous.actions import uniform_grid, environment_action
from muzero_continuous.model import MuZeroModel
from muzero_continuous.targets import value_target


class ActionsTests(unittest.TestCase):
    def test_pendulum_grid_and_format(self):
        grid=uniform_grid(-2,2)
        np.testing.assert_array_equal(grid,[-2,-1,0,1,2])
        for i,expected in enumerate(grid):
            a=environment_action(grid,i)
            self.assertEqual(a.shape,(1,))
            self.assertEqual(a.dtype,np.float32)
            self.assertEqual(a[0],expected)

    def test_reject_invalid_action_and_grid(self):
        for args in ((2,-2,5),(-2,2,1),(float('nan'),2,5)):
            with self.assertRaises(ValueError):
                uniform_grid(*args)
        with self.assertRaises(ValueError):
            environment_action(uniform_grid(-1,1),-1)


class ModelTests(unittest.TestCase):
    def test_batch_shapes_both_environments(self):
        for obs_size,scale in ((3,2),(2,1)):
            model=MuZeroModel(obs_size,5,action_scale=scale)
            first=model.initial_inference(torch.zeros(4,obs_size))
            nxt=model.recurrent_inference(first.hidden,torch.full((4,1),0.37))
            self.assertEqual(first.hidden.shape,(4,32))
            self.assertEqual(nxt.policy_logits.shape,(4,5))
            self.assertEqual(nxt.reward.shape,(4,))
            self.assertEqual(nxt.value.shape,(4,))
            self.assertIsNone(first.reward)
            self.assertTrue(torch.isfinite(nxt.hidden).all())

    def test_numeric_action_reaches_dynamics(self):
        torch.manual_seed(42)
        model=MuZeroModel()
        hidden=torch.zeros(1,32)
        a=torch.tensor([[0.37]],requires_grad=True)
        result=model.recurrent_inference(hidden,a)
        (result.hidden.sum()+result.reward.sum()).backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()),0)

    def test_gradient_through_unroll(self):
        torch.manual_seed(1)
        model=MuZeroModel()
        first=model.initial_inference(torch.ones(2,3))
        nxt=model.recurrent_inference(first.hidden,torch.ones(2,1))
        loss=nxt.value.square().mean()+nxt.reward.square().mean()+nxt.policy_logits.square().mean()
        loss.backward()
        for block in (model.representation,model.dynamics_body,model.policy,model.value,model.reward):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters()))


class TargetsTests(unittest.TestCase):
    def test_real_termination_removes_bootstrap(self):
        self.assertAlmostEqual(value_target([1,2],[False,True],[False,False],100,0.5),2)

    def test_truncation_keeps_bootstrap(self):
        self.assertAlmostEqual(value_target([1,2],[False,False],[False,True],8,0.5),4)

    def test_short_fragment_keeps_bootstrap(self):
        self.assertAlmostEqual(value_target([1],[False],[False],8,0.5),5)

    def test_cannot_cross_episode_boundary(self):
        with self.assertRaises(ValueError):
            value_target([1,2],[False,False],[True,False],8,0.5)


if __name__ == "__main__":
    unittest.main()
