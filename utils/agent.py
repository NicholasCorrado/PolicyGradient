import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(input_dim, output_dim, hidden_dims, activation=nn.Tanh, output_std=0.01):
    layers = []

    # First layer
    layers.append(layer_init(nn.Linear(input_dim, hidden_dims[0])))
    layers.append(activation())

    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(layer_init(nn.Linear(hidden_dims[i], hidden_dims[i + 1])))
        layers.append(activation())

    # Output layer
    layers.append(layer_init(nn.Linear(hidden_dims[-1], output_dim), std=output_std))

    return nn.Sequential(*layers)


class BaseAgent(nn.Module):
    def __init__(self, envs, hidden_dims=None):
        super().__init__()

        # Default to [64, 64] if hidden_dims is None
        self.hidden_dims = [64, 64] if hidden_dims is None else hidden_dims

        # Create critic network (same for all agent types)
        input_dim = self._get_input_dim(envs)
        self.critic = build_mlp(input_dim, 1, self.hidden_dims, output_std=1.0)

    def _get_input_dim(self, envs):
        raise NotImplementedError

    def get_value(self, x):
        return self.critic(x)


class DiscreteActionAgent(BaseAgent):
    def __init__(self, envs, hidden_dims=None):
        self.output_dim = envs.single_action_space.n
        super().__init__(envs, hidden_dims)

        input_dim = self._get_input_dim(envs)
        self.actor = build_mlp(input_dim, self.output_dim, self.hidden_dims, output_std=0.01)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # print(probs.probs)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x, sample=True):
        logits = self.actor(x)
        if sample:
            probs = Categorical(logits=logits)
            # print(probs.probs)
            return probs.sample()
        else:
            return torch.argmax(logits, dim=1)

    def get_action_and_info(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), logits, None

    def get_action_distribution(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs

    def get_logprob(self, x, action):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action)

    def get_actor_parameters(self):
        return self.actor.parameters()


class ContinuousActionAgent(BaseAgent):
    def __init__(self, envs, hidden_dims=None):
        self.output_dim = np.prod(envs.single_action_space.shape)
        super().__init__(envs, hidden_dims)

        input_dim = self._get_input_dim(envs)
        self.actor_mean = build_mlp(input_dim, self.output_dim, self.hidden_dims, output_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.output_dim))

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x, sample=True):
        action_mean = self.actor_mean(x)

        if not sample:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_info(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), action_mean, action_std

    def get_logprob(self, x, action):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1)

    def get_actor_parameters(self):
        return list(self.actor_mean.parameters()) + list(self.actor_logstd)


class DiscreteObsDiscreteActionAgent(DiscreteActionAgent):
    def _get_input_dim(self, envs):
        return envs.single_observation_space.n

    def get_value(self, x):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_value(one_hot_x)

    def get_action_and_value(self, x, action=None):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action_and_value(one_hot_x, action)

    def get_action(self, x, sample=True):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action(one_hot_x, sample)

    def get_action_and_info(self, x, action=None):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action_and_info(one_hot_x, action)

    def get_logprob(self, x, action):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_logprob(one_hot_x, action)


class DiscreteObsContinuousActionAgent(ContinuousActionAgent):
    def _get_input_dim(self, envs):
        return envs.single_observation_space.n

    def get_value(self, x):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_value(one_hot_x)

    def get_action_and_value(self, x, action=None):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action_and_value(one_hot_x, action)

    def get_action(self, x, sample=True):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action(one_hot_x, sample)

    def get_action_and_info(self, x, action=None):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_action_and_info(one_hot_x, action)

    def get_logprob(self, x, action):
        one_hot_x = F.one_hot(x.long(), self._get_input_dim(None)).float()
        return super().get_logprob(one_hot_x, action)


class ContinuousObsDiscreteActionAgent(DiscreteActionAgent):
    def _get_input_dim(self, envs):
        return np.array(envs.single_observation_space.shape).prod()


class ContinuousObsContinuousActionAgent(ContinuousActionAgent):
    def _get_input_dim(self, envs):
        return np.array(envs.single_observation_space.shape).prod()

def Agent(envs, hidden_dims=None):
    """Factory function that returns the appropriate agent based on environment spaces."""
    is_discrete_obs = isinstance(envs.single_observation_space, gym.spaces.Discrete)
    is_discrete_action = isinstance(envs.single_action_space, gym.spaces.Discrete)

    if is_discrete_obs and is_discrete_action:
        return DiscreteObsDiscreteActionAgent(envs, hidden_dims)
    elif is_discrete_obs and not is_discrete_action:
        return DiscreteObsContinuousActionAgent(envs, hidden_dims)
    elif not is_discrete_obs and is_discrete_action:
        return ContinuousObsDiscreteActionAgent(envs, hidden_dims)
    else:  # Continuous observations, continuous actions
        return ContinuousObsContinuousActionAgent(envs, hidden_dims)


class TabularSoftmaxAgent(DiscreteActionAgent):
    """
    Tabular policy agent using softmax policy with one parameter per state-action pair.
    Works with discrete observation and action spaces.
    """

    def __init__(self, envs, hidden_dims=None):
        self.output_dim = envs.single_action_space.n
        BaseAgent.__init__(self, envs, hidden_dims)

        # Override the actor network with a tabular parameterization
        self.num_states = envs.single_observation_space.n
        self.num_actions = self.output_dim

        # Create parameters: one for each state-action pair
        # Shape: (num_states, num_actions)
        self.policy_logits = nn.Parameter(torch.zeros(self.num_states, self.num_actions))

        # Initialize with small random values
        nn.init.xavier_uniform_(self.policy_logits, gain=0.01)

    def _get_input_dim(self, envs):
        return envs.single_observation_space.n

    def get_action_and_value(self, x, action=None):
        # Extract the state indices from the one-hot encoding
        if x.dim() > 1 and x.shape[1] > 1:  # Check if one-hot encoded
            state_indices = torch.argmax(x, dim=1)
        else:  # Already indices
            state_indices = x.long().flatten()

        # Get logits for these states
        logits = self.policy_logits[state_indices]
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x, sample=True):
        # Extract the state indices from the one-hot encoding
        if x.dim() > 1 and x.shape[1] > 1:  # Check if one-hot encoded
            state_indices = torch.argmax(x, dim=1)
        else:  # Already indices
            state_indices = x.long().flatten()

        # Get logits for these states
        logits = self.policy_logits[state_indices]

        if sample:
            probs = Categorical(logits=logits)
            return probs.sample()
        else:
            return torch.argmax(logits, dim=1)

    def get_action_and_info(self, x, action=None):
        # Extract the state indices from the one-hot encoding
        if x.dim() > 1 and x.shape[1] > 1:  # Check if one-hot encoded
            state_indices = torch.argmax(x, dim=1)
        else:  # Already indices
            state_indices = x.long().flatten()

        # Get logits for these states
        logits = self.policy_logits[state_indices]
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), logits, None

    def get_logprob(self, x, action):
        # Extract the state indices from the one-hot encoding
        if x.dim() > 1 and x.shape[1] > 1:  # Check if one-hot encoded
            state_indices = torch.argmax(x, dim=1)
        else:  # Already indices
            state_indices = x.long().flatten()

        # Get logits for these states
        logits = self.policy_logits[state_indices]
        probs = Categorical(logits=logits)

        return probs.log_prob(action)