import torch


def compute_second_order_policy_gradient(agent, obs, actions, returns, damping=1e-3, cg_iters=10):
    """Compute natural policy gradient for the given transitions

    Args:
        agent: The agent containing actor network
        obs: Batch of observations
        actions: Batch of actions taken
        returns: Batch of returns-to-go
        damping: Regularization coefficient for Fisher matrix
        cg_iters: Number of conjugate gradient iterations to approximate F^-1 * g

    Returns:
        natural_policy_gradient: The natural policy gradient vector
    """
    # First compute the regular policy gradient
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:]).long()
    b_returns = returns.reshape(-1)

    # Standardize returns for more stable gradients
    if b_returns.shape[0] > 1:
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

    # Get first-order policy gradient
    agent.zero_grad()
    log_probs = agent.get_logprob(b_obs, b_actions)
    policy_loss = -(log_probs * b_returns).mean()
    policy_loss.backward()

    # Extract and concatenate gradients
    policy_gradient = []
    for param in agent.actor.parameters():
        if param.grad is not None:
            policy_gradient.append(param.grad.clone().view(-1))

    policy_gradient = torch.cat(policy_gradient)
    agent.zero_grad()

    # Function to compute Fisher-vector product (F*x)
    def fisher_vector_product(v):
        # Ensure v has the right shape and convert to a PyTorch tensor if needed
        v = torch.FloatTensor(v).to(b_obs.device)

        # Compute the second derivative term: F * v
        # First pass: compute kl divergence
        kl = compute_kl_divergence(agent, b_obs)

        # Compute the product of Hessian of KL and vector v
        grads = torch.autograd.grad(
            kl,
            agent.actor.parameters(),
            create_graph=True,
            retain_graph=True
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        # Compute inner product of gradient and vector
        kl_v = (flat_grad_kl * v).sum()

        # Compute the gradient w.r.t. model parameters
        grads = torch.autograd.grad(kl_v, agent.actor.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl + damping * v

    # Use conjugate gradient algorithm to compute x = F^-1 * g
    natural_policy_gradient = conjugate_gradient(
        fisher_vector_product,
        policy_gradient,
        cg_iters
    )

    return natural_policy_gradient


def compute_kl_divergence(agent, states):
    """Compute the KL divergence between current policy and a reference policy"""
    # Get the distribution parameters of the current policy
    with torch.no_grad():
        ref_dist = agent.get_action_distribution(states)

    # Get the distribution parameters of the reference policy
    curr_dist = agent.get_action_distribution(states)

    # Compute KL divergence between the two distributions
    kl = torch.distributions.kl.kl_divergence(ref_dist, curr_dist).mean()

    return kl


def conjugate_gradient(A_fn, b, num_iters=10, residual_tol=1e-10):
    """
    Use conjugate gradient algorithm to solve Ax = b

    Args:
        A_fn: Function that computes matrix-vector product Ax given x
        b: Vector in the linear system Ax = b
        num_iters: Number of CG iterations
        residual_tol: Tolerance for early stopping

    Returns:
        x: Solution to Ax = b
    """
    x = torch.zeros_like(b)
    r = b.clone()  # Residual
    p = r.clone()  # Search direction

    for i in range(num_iters):
        Ap = A_fn(p)
        alpha = torch.dot(r, r) / torch.dot(p, Ap)

        x = x + alpha * p
        r_new = r - alpha * Ap

        if torch.norm(r_new) < residual_tol:
            break

        beta = torch.dot(r_new, r_new) / torch.dot(r, r)
        p = r_new + beta * p
        r = r_new

    return x