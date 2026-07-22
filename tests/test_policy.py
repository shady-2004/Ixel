import random
from policy import BanditPolicy


def test_policy_learning():
    policy = BanditPolicy(input_dim=1, lr=0.1, epsilon=0.2)
    for _ in range(10):
        feat = [random.random()]
        action, q_values = policy.select_action(feat, explore=True)
        assert action in ["INDEX", "NO_ACTION", "DROP"]
        reward = 1.0
        loss = policy.update(feat, action, reward)
        assert isinstance(loss, float)
        
    action_eval, _ = policy.select_action([0.9], explore=False)
    assert action_eval in ["INDEX", "NO_ACTION", "DROP"]
