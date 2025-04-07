import torch


def compute_first_order_policy_gradient(agent, obs, actions, returns):
    """Compute policy gradient for the given transitions"""
    # Make sure to zero out all gradients
    agent.zero_grad()

    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:]).long()
    b_returns = returns.reshape(-1)

    # # Standardize returns for more stable gradients
    # if b_returns.shape[0] > 1:  # Only standardize if we have more than one sample
    #     b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
    #     # b_returns = (b_returns - b_returns.mean())

    log_probs = agent.get_logprob(b_obs, b_actions)
    policy_loss = -(log_probs * b_returns).mean()

    policy_loss.backward()

    # Extract and concatenate gradients
    policy_gradient = []
    for param in agent.get_actor_parameters():
        if param.grad is not None:
            policy_gradient.append(param.grad.clone().view(-1))  # Use clone() to avoid reference issues

    # Clear gradients immediately to prevent any potential leakage
    agent.zero_grad()

    return torch.cat(policy_gradient)