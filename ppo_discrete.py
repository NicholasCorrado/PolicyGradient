# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import custom_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro as tyro
import yaml

# import custom_envs
from stable_baselines3.common.utils import get_latest_run_id
from torch.distributions.categorical import Categorical

from collections import deque
from actor_update import ppo_update, ppo_update_value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, linear=False):
        super().__init__()
        obs_dim = np.prod(envs.single_observation_space.shape)
        action_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        if linear:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, action_dim), std=0.01),
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_logprob_and_value(self, x, action):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action), self.critic(x)

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def get_action_and_info(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_logprob(self, x, action):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action)

    def reset_actor(self):
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                layer_init(layer)
        layer_init(self.actor[-1], std=0.01)

    def get_probs(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.probs


def make_env(env_id, idx, capture_video=False, run_name="run"):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def simulate(env, actor, eval_episodes, eval_steps=np.inf, sample=False):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'))
                actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            logs_episode['rewards'].append(rewards)
            logs_episode['actions'].append(actions.item())

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break

        logs['returns'].append(np.sum(logs_episode['rewards']))
        logs['actions'].extend(logs_episode['actions'])
        # print(actions)


        try:
            logs['successes'].append(infos['final_info'][0]['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    actions = np.sum(np.array(logs['actions']) == 0)
    print(actions)
    return return_avg, return_std, success_avg, success_std

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_policy: bool = False
    load_policy_path: str = ''

    # Logging
    output_rootdir: str = 'results'
    output_subdir: str = ''
    run_id: int = None
    seed: int = None
    total_timesteps: int = 1000000

    # Evaluation
    num_evals: int = None
    eval_freq: int = 10
    eval_episodes: int = 100
    compute_sampling_error: bool = False

    # Architecture arguments
    linear: int = 1
    actor_init_std: float = 0.01

    # Learning algorithm
    algo: str = 'ppo'

    # Sampling algorithm
    # sampling_algo: str = 'props'
    sampling_algo: str = 'on_policy'

    # Algorithm specific arguments
    env_id: str = "GridWorld-5x5-v0"
    learning_rate: float = 1e-3
    num_envs: int = 1
    num_steps: int = 32
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    buffer_batches: int = 1
    warm_start_steps: int = 100

    # Behavior
    props_num_steps: int = 16
    props_learning_rate: float = 1e-3
    props_update_epochs: int = 16
    props_num_minibatches: int = 4
    props_clip_coef: float = 0.3
    props_target_kl: float = 0.01
    props_lambda: float = 0.0
    props_freeze_features: bool = False

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def run():
    args = tyro.cli(Args)
    args.buffer_size = args.buffer_batches * args.num_steps

    args.batch_size = int(args.num_envs * args.buffer_size)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.props_batch_size = int(args.num_envs * (args.buffer_size - args.props_num_steps))
    args.props_minibatch_size = int(args.props_batch_size // args.props_num_minibatches)

    args.num_iterations = args.total_timesteps // args.num_steps
    if args.num_evals:
        args.eval_freq = max(args.num_iterations // args.num_evals, 1)
    # props_iterations_per_update = args.num_steps // args.props_num_steps

    if args.sampling_algo in ['props', 'ros']:
        assert args.num_steps % args.props_num_steps == 0

    ### Seeding
    if args.run_id:
        args.seed = args.run_id
    elif args.seed is None:
        args.seed = np.random.randint(2 ** 32 - 1)

    ### Override hyperparameters based on sampling method
    assert args.sampling_algo in ['on_policy', 'ros', 'props', 'greedy_adaptive', 'oracle_adaptive']
    if args.algo == 'ros':
        args.props_num_steps = 1

    ### Output path
    args.output_dir = f"{args.output_rootdir}/{args.env_id}/{args.algo}/{args.sampling_algo}/{args.output_subdir}"
    if args.run_id is not None:
        args.output_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(log_path=args.output_dir, log_name='run_') + 1
        args.output_dir += f"/run_{run_id}"

    ### Dump training config to save dir
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    ### wandb
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=args.output_dir,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    env_eval = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(1)],
    )
    envs_se = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(1)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    if args.load_policy_path:
        agent = torch.load(args.load_policy_path, weights_only=False)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # optimizer = optim.SGD(agent.parameters(), lr=args.learning_rate)

    agent_props = copy.deepcopy(agent)

    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    if args.props_freeze_features:
        params = [p for p in agent_props.actor.parameters()]
        for p in params[:4]:
            p.requires_grad = False

    optimizer_props = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(device)

    # buffer_pos = 0
    # global_step = 0
    # if args.collect_before_training:
    #     # collect(obs_buffer, actions_buffer, rewards_buffer, dones_buffer)
    #     # agent_collect = torch.load(args.collect_policy_path, weights_only=False)
    #     agent_collect = Agent(envs, args.linear).to(device)
    #     fill_buffers(agent_collect, envs, obs_buffer, actions_buffer, args.collect_before_training, device)
    #     buffer_pos = args.collect_before_training
    #     global_step = args.collect_before_training

    # for computing sampling error during RL
    agent_buffer = deque(maxlen=args.buffer_batches)
    envs_buffer = deque(maxlen=args.buffer_batches)
    obs_buffer_se = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer_se = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)

    ### Logging
    logs = defaultdict(list)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    buffer_pos = 0
    update_count = 0
    eval_count = 0
    start_time = time.time()

    # # Eval at t=0
    # return_avg, return_std, success_avg, success_std = simulate(env=env_eval, actor=agent,eval_episodes=args.eval_episodes)
    # print(
    #     f"Eval num_timesteps={global_step}, " f"episode_return={return_avg:.2f} +/- {return_std:.2f}\n"
    #     f"Eval num_timesteps={global_step}, " f"episode_success={success_avg:.2f} +/- {success_std:.2f}\n"
    # )
    # logs['timestep'].append(global_step)
    # logs['return'].append(return_avg)
    # logs['success_rate'].append(success_avg)
    # logs['update'].append(update_count)

    if args.save_policy:
        torch.save(agent.state_dict(), f"{args.output_dir}/policy_{eval_count}.pt")

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.warm_start_steps > 0:
        print('warm starting value function...')

        # obs_buffer_ws = torch.zeros((args.warm_start_steps*envs.envs[0]._max_episode_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        # rewards_buffer_ws = torch.zeros((args.warm_start_steps*envs.envs[0]._max_episode_steps, args.num_envs)).to(device)
        # dones_buffer_ws = torch.zeros((args.warm_start_steps*envs.envs[0]._max_episode_steps, args.num_envs)).to(device)

        obs_buffer_ws = []
        rewards_buffer_ws = []
        dones_buffer_ws = []

        num_traj = 0
        step = 0
        while num_traj < args.warm_start_steps:
        # for step in range(0, args.warm_start_steps):
            obs_buffer_ws.append(next_obs)
            dones_buffer_ws.append(next_done)

            with torch.no_grad():
                action = agent.get_action(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards_buffer_ws.append(torch.tensor(reward).to(device).view(-1))
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            step += 1
            global_step += 1

            if next_done:
                num_traj += 1

        print(step)
        obs = torch.stack(obs_buffer_ws)
        rewards = torch.tensor(rewards_buffer_ws)
        dones = torch.tensor(dones_buffer_ws)
        with torch.no_grad():
            values = agent.get_value(obs).reshape(-1, args.num_envs)
        #
        # with torch.no_grad():
        #     next_value = agent.get_value(next_obs).reshape(1, -1)
        #     advantages = torch.zeros_like(rewards).to(device)
        #     lastgaelam = 0
        #     for t in reversed(range(args.warm_start_steps)):
        #         if t == args.warm_start_steps - 1:
        #             nextnonterminal = 1.0 - next_done
        #             nextvalues = next_value
        #         else:
        #             nextnonterminal = 1.0 - dones[t + 1]
        #             nextvalues = values[t + 1]
        #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        #     returns = advantages + values

        num_samples = len(obs)
        with torch.no_grad():
            next_value = 0
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_samples)):
                if t == num_samples - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages

        # flatten buffer data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        optimizer_ws = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        ppo_update_value(agent, optimizer_ws, b_obs, b_returns, b_values, args)

        args.num_iterations = (args.total_timesteps)// args.num_steps - (num_samples)// args.num_steps
        if args.num_evals:
            args.eval_freq = max(args.num_iterations // args.num_evals, 1)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # if iteration % args.eval_freq == 0 and args.compute_sampling_error:
            # agent_buffer.appendleft(copy.deepcopy(agent))
            # envs_buffer.appendleft(copy.deepcopy(envs))
        agent_buffer.append(copy.deepcopy(agent))
        envs_buffer.append(copy.deepcopy(envs))

        # if global_step > args.buffer_size:
        #     # shift buffers left by one batch. We will place the next batch we collect at the end of the buffer.
        #     obs_buffer = torch.roll(obs_buffer, shifts=-args.num_steps, dims=0)
        #     actions_buffer = torch.roll(actions_buffer, shifts=-args.num_steps, dims=0)
        #     rewards_buffer = torch.roll(rewards_buffer, shifts=-args.num_steps, dims=0)
        #     dones_buffer = torch.roll(dones_buffer, shifts=-args.num_steps, dims=0)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_buffer[buffer_pos] = next_obs
            dones_buffer[buffer_pos] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.sampling_algo in ['props', 'ros']:
                    action = agent_props.get_action(next_obs)
                else:
                    action = agent.get_action(next_obs)

            # n_actions = envs.single_action_space.n
            # action = torch.Tensor([np.random.choice(a=np.arange(1, n_actions), p=np.ones(n_actions-1)/(n_actions-1))]).type(torch.int)
            # action[:] = global_step % 99 + 1
            actions_buffer[buffer_pos] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards_buffer[buffer_pos] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Once the buffer is full, we only write to last batch in the buffer for all subsequent collection phases.
            buffer_pos += 1
            # buffer_pos = buffer_pos % args.buffer_size
            if buffer_pos == args.buffer_size:
                buffer_pos = args.buffer_size - args.num_steps

            ################################## START BEHAVIOR UPDATE ##################################
            log_props = {}
            if args.sampling_algo in ['props', 'ros'] and global_step % args.props_num_steps == 0: # and global_step >= args.num_steps:

                end = buffer_pos if buffer_pos > 0 else args.buffer_size
                obs = obs_buffer[:end]
                actions = actions_buffer[:end]

                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape).long()
                with torch.no_grad():
                    logprobs = agent.get_logprob(b_obs, b_actions)

                b_logprobs = logprobs.reshape(-1)

                for source_param, dump_param in zip(agent_props.parameters(), agent.parameters()):
                    source_param.data.copy_(dump_param.data)

                if args.sampling_algo == 'props':
                    log_props = props_update(agent_props, optimizer_props, b_obs, b_actions, b_logprobs, args)
                elif args.sampling_algo == 'ros':
                    log_props = ros_update(agent_props, optimizer_props, b_obs, b_actions, b_logprobs, args)

            ################################## END BEHAVIOR UPDATE ##################################

        obs = obs_buffer[:global_step]
        actions = actions_buffer[:global_step]
        rewards = rewards_buffer[:global_step]
        dones = dones_buffer[:global_step]

        with torch.no_grad():
            values = agent.get_value(obs).reshape(-1, args.num_envs)
            logprobs = agent.get_logprob(obs, actions).reshape(-1, args.num_envs)


        # values[:] = 0
        # bootstrap value if not done
        # @TODO: make sure this runs over the most recent batch
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        ### Target policy (and value network) update
        # flatten buffer data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape).long()
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        log_target = {}
        if args.learning_rate > 0 and args.update_epochs > 0:
            update_count += 1
            log_target = ppo_update(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args)
        ################################## END TARGET UPDATE ##################################

        if iteration % args.eval_freq == 0:
            eval_count += 1
            return_avg, return_std, success_avg, success_std = simulate(env=env_eval, actor=agent, eval_episodes=args.eval_episodes, sample=True)
            print(
                f"Eval num_timesteps={global_step}, " f"episode_return={return_avg:.2f} +/- {return_std:.2f}\n"
                f"Eval num_timesteps={global_step}, " f"episode_success={success_avg:.2f} +/- {success_std:.2f}\n"
            )

            logs['timestep'].append(global_step)
            logs['return'].append(return_avg)
            logs['success_rate'].append(success_avg)
            logs['update'].append(update_count)
            for key, value in log_props.items():
                logs[key].append(value)
            for key, value in log_target.items():
                logs[key].append(value)

            np.savez(f'{args.output_dir}/evaluations.npz', **logs)

            if args.track:
                log_wandb = {}
                for key, value in logs.items():
                    log_wandb[key] = value[-1]
                wandb.log(log_wandb)


            if args.save_policy:
                torch.save(agent.state_dict(), f"{args.output_dir}/policy_{eval_count}.pt")

    envs.close()
    # writer.close()




if __name__ == "__main__":
    run()