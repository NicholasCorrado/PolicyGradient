import os

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/example.txt", "w")

    total_timesteps = int(50e3)
    for env_id in [
        # 'GridWorld-5x5-v0'
        'CartPole-v1',
        'LunarLander-v3',
        'Discrete2D100-v0'
    ]:
        for lr in [3e-4]:
            for ns in [512, 1024, 2048]:
                command = (
                    f"python ppo_discrete.py --output_subdir lr_{lr}/ns_{ns}"
                    f" --env_id {env_id}"
                    f" --learning_rate {lr}"
                    f" --num_steps {ns}"
                    f" --total_timesteps {total_timesteps}"
                    f" --eval_freq 10"
                    f" --eval_episodes 100"
                    f" --sampling_algo on_policy"
                )

                for i in range(0, 20):
                    for ws in [0, 16]:
                        total_timesteps = 1024 * 8

                        command = (
                            f"python ../ppo_discrete.py --run_id {i} --seed {i}"
                            f" --env_id {env_id}"
                            f" --total_timesteps {total_timesteps} --eval_freq 10"
                            f" --learning_rate {lr} --num_steps {ns}"
                            f" --output_rootdir results"
                            f" --output_subdir  ns_{ns}/ws_{ws}"
                            f" --warm_start_steps {ws}"
                            f" --eval_freq 100"
                            f" --update_epochs 1"
                            f" --num_minibatches 1"
                        )

                mem = 0.5
                disk = 1
                command = f"{mem},{disk},{command}"
                print(command)
                f.write(command + "\n")
