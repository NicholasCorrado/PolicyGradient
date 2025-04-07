import torch


def compute_grad_norms_vectorized(agent, log_probs_with_grad):
    """
    Compute gradient norms for log probabilities with minimal looping

    Args:
        agent: The policy agent
        log_probs_with_grad: Log probabilities with gradients attached

    Returns:
        torch.Tensor: Vector of squared gradient norms
    """
    batch_size = log_probs_with_grad.shape[0]
    grad_norms_squared = torch.zeros(batch_size, device=log_probs_with_grad.device)

    # Process examples in a single batch where possible
    # For large batches, we can process in chunks to reduce memory usage
    chunk_size = min(16, batch_size)  # Adjust based on your GPU memory

    for chunk_start in range(0, batch_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, batch_size)

        # Process this chunk of examples
        # for i in range(chunk_start, chunk_end):
        #     # Get gradients for this individual example
        #     example_grads = torch.autograd.grad(
        #         log_probs_with_grad[i],
        #         agent.actor.parameters(),
        #         retain_graph=(i < batch_size - 1),  # Only retain graph if not the last item
        #         create_graph=False,
        #         allow_unused=True
        #     )

            # Get gradients for this individual example
        example_grads = torch.autograd.grad(
            log_probs_with_grad[chunk_start:chunk_end].mean(),
            agent.actor.parameters(),
            retain_graph=True,  # Only retain graph if not the last item
            create_graph=False,
            allow_unused=True
        )

        # Sum the squared gradients in a vectorized way
        grad_norm_squared = 0.0
        for grad in example_grads:
            if grad is not None:
                # Use .sum() to vectorize the squaring and summing operation
                grad_norm_squared += torch.norm(grad)**2

        for i in range(chunk_start, chunk_end):

            grad_norms_squared[i] = grad_norm_squared

    return grad_norms_squared

def compute_policy_gradient_min_variance(agent, obs, actions, returns):
    """
    Compute policy gradient for the given transitions using the optimal baseline
    that minimizes the variance of gradient estimates.

    The optimal baseline depends on the squared norm of the policy gradient: ||∇log π||²
    b* = E[R * ||∇log π||²] / E[||∇log π||²]
    """
    import torch

    # Reshape inputs
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:]).long()
    b_returns = returns.reshape(-1)

    grad_norms_squared = torch.zeros(len(b_obs))
    agent.zero_grad()

    for i in range(len(b_obs)):
        log_probs = -agent.get_logprob(b_obs[i].view(1, -1), b_actions[i].view(1, -1))
        log_probs.sum().backward()

        # grad = [param.grad for param in agent.actor_mean] + [param.grad for param in agent.actor_logstd]

        for param in agent.actor.parameters():
            if param.grad is not None:
                grad_norms_squared[i] += param.grad.pow(2).sum()

        # Clear gradients immediately to prevent any potential leakage
        agent.zero_grad()

    log_probs_with_grad = agent.get_logprob(b_obs, b_actions)

    # # Initialize tensor to store gradient norms
    # grad_norms_squared = torch.zeros_like(b_returns)
    #
    # Calculate gradient norms squared for each parameter and sample
    # We'll use vectorized operations where possible
    # for i in range(log_probs_with_grad.shape[0]):
    #     # Compute gradient of this single log prob w.r.t model parameters
    #     grad_params = torch.autograd.grad(
    #         log_probs_with_grad[i],
    #         agent.actor.parameters(),
    #         retain_graph=(i < log_probs_with_grad.shape[0] - 1),  # Only retain graph if not the last item
    #         create_graph=False,
    #         allow_unused=True
    #     )
    #
    #     # Sum up squared norms of gradients
    #     grad_norm_squared = 0.0
    #     for grad in grad_params:
    #         if grad is not None:
    #             grad_norm_squared += grad.pow(2).sum().item()
    #
    #     grad_norms_squared[i] = grad_norm_squared

    grad_norms_squared = compute_grad_norms_vectorized(agent, log_probs_with_grad)
    # Release memory
    b_obs.requires_grad_(False)
    del log_probs_with_grad

    # Calculate optimal baseline
    # b* = E[R * ||∇log π||²] / E[||∇log π||²]
    numerator = (b_returns * grad_norms_squared).mean()
    denominator = grad_norms_squared.mean() + 1e-8
    optimal_baseline = numerator / denominator

    # Calculate advantages
    advantages = b_returns - optimal_baseline

    # Now compute the final policy gradient
    agent.zero_grad()

    # Recompute log probabilities and calculate the loss
    log_probs = agent.get_logprob(b_obs, b_actions)
    policy_loss = -(log_probs * advantages).mean()

    # Backpropagate
    policy_loss.backward()

    # Extract and concatenate gradients
    policy_gradient = []
    for param in agent.actor.parameters():
        if param.grad is not None:
            policy_gradient.append(param.grad.clone().view(-1))

    # Clear gradients
    agent.zero_grad()

    return torch.cat(policy_gradient)




def compute_policy_gradient_new(agent, obs, actions, returns, num_bins=20):
    """
    Compute policy gradient with gradient-weighted baseline for variance reduction.

    Args:
        agent: The policy agent
        obs: Batch of observations (states)
        actions: Batch of actions taken
        returns: Batch of returns
        num_bins: Number of bins to discretize action probabilities

    Returns:
        torch.Tensor: Policy gradient
    """
    # Make sure to zero out all gradients
    agent.zero_grad()

    # Reshape inputs
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:]).long()
    b_returns = returns.reshape(-1)

    # 1. Discretize action probability space into k evenly spaced bins
    # Create bins from 0 to 1
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2

    # 2. Compute ||\nabla log p||^2 for each bin midpoint
    grad_lookup = {}
    for p in bin_midpoints:
        # Create a dummy probability that requires gradient
        dummy_prob = torch.tensor(p, requires_grad=True)
        log_p = torch.log(dummy_prob)
        log_p.backward()
        # Get the squared gradient norm
        grad_norm_squared = (dummy_prob.grad ** 2).item()
        # Store in lookup table
        grad_lookup[p.item()] = grad_norm_squared

    # 3-4. Compute probabilities for each state-action pair and map to bins
    with torch.no_grad():  # Don't accumulate gradients for this computation
        # Get action probabilities for each state-action pair
        action_probs = agent.get_logprob(b_obs, b_actions).exp()

        # Find which bin each probability falls into
        w = torch.zeros_like(b_returns)
        for i, prob in enumerate(action_probs):
            # Find the nearest bin midpoint
            bin_idx = torch.abs(bin_midpoints - prob).argmin().item()
            nearest_midpoint = bin_midpoints[bin_idx].item()
            # Assign the corresponding gradient norm
            w[i] = grad_lookup[nearest_midpoint]

    # Reset gradients before actual backward pass
    agent.zero_grad()

    # Get log probabilities for the policy gradient
    log_probs = agent.get_logprob(b_obs, b_actions)

    # 5. Use gradient-weighted baseline
    if w.sum() > 0 and b_returns.shape[0] > 1:
        # Only use weighted baseline if we have valid weights and more than one sample
        baseline = torch.sum(b_returns * w) / torch.sum(w)
        advantages = b_returns - baseline
    else:
        # Fall back to regular standardization if weights are zero or we have only one sample
        advantages = b_returns
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute policy loss
    policy_loss = -(log_probs * advantages).mean()

    # Backward pass
    policy_loss.backward()

    # Extract and concatenate gradients
    policy_gradient = []
    for param in agent.actor.parameters():
        if param.grad is not None:
            policy_gradient.append(param.grad.clone().view(-1))

    # Clear gradients immediately to prevent any potential leakage
    agent.zero_grad()

    return torch.cat(policy_gradient)
