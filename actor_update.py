import numpy as np
import torch
from torch import nn

def ppo_update(
        agent,  # Policy network (actor-critic architecture)
        optimizer,  # Optimizer (typically Adam)
        b_obs,  # Batch of observations (states)
        b_actions,  # Batch of actions taken
        b_logprobs,  # Batch of log probabilities of taken actions
        b_advantages,  # Batch of advantage estimates
        b_returns,  # Batch of returns (discounted rewards)
        b_values,  # Batch of value estimates
        args,  # Arguments containing hyperparameters
):
    """
    Performs a PPO policy update step using minibatching and clipped objectives.

    Args:
        agent: The actor-critic policy network that is being updated
        optimizer: The optimizer (typically Adam) used for updating the policy
        b_obs (torch.Tensor): Batch of observations/states from the environment
        b_actions (torch.Tensor): Batch of actions taken in the environment
        b_logprobs (torch.Tensor): Log probabilities of the actions taken under the old policy
        b_advantages (torch.Tensor): Computed advantage estimates for each timestep
        b_returns (torch.Tensor): Computed returns (discounted sum of rewards)
        b_values (torch.Tensor): Value estimates from the old policy
        args: Object containing PPO hyperparameters including:
            - num_minibatches (int): Number of minibatches to split the data into
            - update_epochs (int): Number of epochs to update on the same batch of data
            - clip_coef (float): PPO clipping coefficient (epsilon in the paper)
            - norm_adv (bool): Whether to normalize advantages
            - clip_vloss (bool): Whether to use clipped value loss
            - ent_coef (float): Entropy bonus coefficient
            - vf_coef (float): Value function loss coefficient
            - max_grad_norm (float): Maximum gradient norm for clipping
            - target_kl (float, optional): Target KL divergence threshold for early stopping
        target_update_count (int, optional): Counter for tracking target network updates
    """
    ### Target policy (and value network) update
    batch_size = len(b_obs)
    minibatch_size = max(batch_size // args.num_minibatches, 1)
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    logs = {
        'ppo/policy_loss': pg_loss.item(),
        'ppo/entropy_loss': entropy_loss.item(),
        'ppo/approx_kl': approx_kl.item(),
        'ppo/clipfrac': np.mean(clipfracs),
        'ppo/epoch': epoch
    }
    return logs

def ppo_update_value(
        agent,  # Policy network (actor-critic architecture)
        optimizer,  # Optimizer (typically Adam)
        b_obs,  # Batch of observations (states)
        b_returns,  # Batch of returns (discounted rewards)
        b_values,  # Batch of value estimates
        args,  # Arguments containing hyperparameters
):
    """
    Performs a PPO policy update step using minibatching and clipped objectives.

    Args:
        agent: The actor-critic policy network that is being updated
        optimizer: The optimizer (typically Adam) used for updating the policy
        b_obs (torch.Tensor): Batch of observations/states from the environment
        b_actions (torch.Tensor): Batch of actions taken in the environment
        b_logprobs (torch.Tensor): Log probabilities of the actions taken under the old policy
        b_advantages (torch.Tensor): Computed advantage estimates for each timestep
        b_returns (torch.Tensor): Computed returns (discounted sum of rewards)
        b_values (torch.Tensor): Value estimates from the old policy
        args: Object containing PPO hyperparameters including:
            - num_minibatches (int): Number of minibatches to split the data into
            - update_epochs (int): Number of epochs to update on the same batch of data
            - clip_coef (float): PPO clipping coefficient (epsilon in the paper)
            - norm_adv (bool): Whether to normalize advantages
            - clip_vloss (bool): Whether to use clipped value loss
            - ent_coef (float): Entropy bonus coefficient
            - vf_coef (float): Value function loss coefficient
            - max_grad_norm (float): Maximum gradient norm for clipping
            - target_kl (float, optional): Target KL divergence threshold for early stopping
        target_update_count (int, optional): Counter for tracking target network updates
    """
    ### Target policy (and value network) update
    batch_size = len(b_obs)
    minibatch_size = max(batch_size // args.num_minibatches, 1)
    minibatch_size = batch_size
    b_inds = np.arange(batch_size)
    # grad_norm = 1
    # while grad_norm > 1e-2:
    for epoch in range(100):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            newvalue = agent.get_value(b_obs[mb_inds])

            # Value loss
            newvalue = newvalue.view(-1)
            # if args.clip_vloss:
            #     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            #     v_clipped = b_values[mb_inds] + torch.clamp(
            #         newvalue - b_values[mb_inds],
            #         -args.clip_coef,
            #         args.clip_coef,
            #     )
            #     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            #     v_loss = 0.5 * v_loss_max.mean()
            # else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            loss = v_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            print(grad_norm)

    logs = {}
    return logs