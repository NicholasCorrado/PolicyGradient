import torch


def compute_third_order_policy_gradient(agent, obs, actions, returns, damping=1e-3, cg_iters=10,
                                        tensor_reg=1e-4, third_order_weight=0.5):
    """Compute third-order policy gradient for the given transitions

    Args:
        agent: The agent containing actor network
        obs: Batch of observations
        actions: Batch of actions taken
        returns: Batch of returns-to-go
        damping: Regularization coefficient for Fisher matrix
        cg_iters: Number of conjugate gradient iterations
        tensor_reg: Regularization for third-order tensor calculations
        third_order_weight: Weight for the third-order correction term

    Returns:
        third_order_policy_gradient: The third-order policy gradient vector
    """
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:]).long()
    b_returns = returns.reshape(-1)

    # Standardize returns for more stable gradients
    if b_returns.shape[0] > 1:
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

    # Step 1: Compute first-order policy gradient
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

    # Step 2: Compute natural policy gradient using conjugate gradient
    def fisher_vector_product(v):
        v = torch.FloatTensor(v).to(b_obs.device)
        kl = compute_kl_divergence(agent, b_obs)

        grads = torch.autograd.grad(
            kl,
            agent.actor.parameters(),
            create_graph=True,
            retain_graph=True
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()

        grads = torch.autograd.grad(kl_v, agent.actor.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_grad_kl + damping * v

    # Get second order natural gradient: F^-1 * g
    natural_policy_gradient = conjugate_gradient(
        fisher_vector_product,
        policy_gradient,
        cg_iters
    )

    # Step 3: Compute third-order correction
    # This computes how the Fisher matrix changes in the direction of the natural gradient
    def third_order_correction(v, u):
        """Compute tensor-vector-vector product: T(v,u)"""
        v = torch.FloatTensor(v).to(b_obs.device)
        u = torch.FloatTensor(u).to(b_obs.device)

        # First compute Fisher-vector product: F * v
        kl = compute_kl_divergence(agent, b_obs)

        grads = torch.autograd.grad(
            kl,
            agent.actor.parameters(),
            create_graph=True,
            retain_graph=True
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()

        # Compute Jacobian-vector product: ∇(F*v)
        grads = torch.autograd.grad(kl_v, agent.actor.parameters(), create_graph=True, retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        # Now compute directional derivative of F*v in direction u: ∇(F*v)*u
        grad_grad_kl_u = (flat_grad_grad_kl * u).sum()

        # Final derivative gives us the tensor-vector-vector product: T(v,u)
        grads = torch.autograd.grad(grad_grad_kl_u, agent.actor.parameters(), retain_graph=True)
        tensor_vector_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return tensor_vector_vector_product + tensor_reg * (v + u)

    # Compute the third-order correction term: 0.5 * F^-1 * T(ng, ng)
    # Where ng is the natural gradient
    tensor_correction = third_order_correction(natural_policy_gradient, natural_policy_gradient)

    # Solve for F^-1 * T(ng, ng) using conjugate gradient
    correction_term = conjugate_gradient(
        fisher_vector_product,
        tensor_correction,
        cg_iters
    )

    # Final third-order policy gradient: natural_gradient - 0.5 * weight * correction_term
    third_order_policy_gradient = natural_policy_gradient - third_order_weight * 0.5 * correction_term

    return third_order_policy_gradient


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
        alpha = torch.dot(r, r) / (torch.dot(p, Ap) + 1e-8)

        x = x + alpha * p
        r_new = r - alpha * Ap

        if torch.norm(r_new) < residual_tol:
            break

        beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)
        p = r_new + beta * p
        r = r_new

    return x


def update_policy_with_third_order_gradient(agent, gradient, learning_rate=0.01):
    """Update the policy parameters using the third-order policy gradient"""
    idx = 0
    for param in agent.actor.parameters():
        num_params = param.numel()
        update = gradient[idx:idx + num_params].view(param.shape)
        param.data += learning_rate * update
        idx += num_params