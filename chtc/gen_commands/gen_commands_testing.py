import os

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/example.txt", "w")

    timesteps = 30000

    buffer_batches = 64
    num_steps = 64
    total_timesteps = buffer_batches * num_steps
    for env_id in [
        'CartPole-v1',
        'LunarLander-v3',
        'Discrete2D100-v0'
    ]:
            # command = (
            #     f"python ppo_discrete.py"
            #     f" --env_id {env_id}"
            #     f" --buffer_batches {buffer_batches}"
            #     f" --num_steps {num_steps}"
            #     f" --learning_rate 0"
            #     f" --total_timesteps {total_timesteps}"
            #     f" --eval_freq 1"
            #     f" --eval_episodes 0"
            #     f" --compute_sampling_error"
            #     f" --sampling_algo on_policy"
            # )
            #
            # mem = 0.3
            # disk = 0.5
            # command = f"{mem},{disk},{command}"
            # print(command)
            # f.write(command + "\n")

            for plr in [1e-2]:
                for pns in [8]:
                    for pkl in [0.03]:
                        for pc in [0.3]:
                            for pe in [32, 64, 128]:
                                command = (
                                    f"python ppo_discrete.py"
                                    f" --env_id {env_id}"
                                    f" --buffer_batches {buffer_batches}"
                                    f" --num_steps {num_steps}"
                                    f" --learning_rate 0"
                                    f" --total_timesteps {total_timesteps}"
                                    f" --eval_freq 1"
                                    f" --eval_episodes 0"
                                    f" --compute_sampling_error"
                                    f" --sampling_algo props "
                                    f" --props_learning_rate {plr}"
                                    f" --props_num_steps {pns}"
                                    f" --props_target_kl {pkl} "
                                    f" --props_clip_coef {pc} "
                                    f" --props_update_epochs {pe}"
                                    f" --output_subdir plr_{plr}/pns_{pns}/pe_{pe}/pkl_{pkl}/pc_{pc}"
                                )

                                mem = 0.3
                                disk = 0.1
                                command = f"{mem},{disk},{command}"
                                print(command)
                                f.write(command + "\n")

            # for plr in [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2]:
            #     command = (
            #         f"python ppo_discrete.py"
            #         f" --env_id {env_id}"
            #         f" --buffer_batches {buffer_batches}"
            #         f" --num_steps {num_steps}"
            #         f" --learning_rate 0"
            #         f" --total_timesteps {total_timesteps}"
            #         f" --eval_freq 1"
            #         f" --eval_episodes 0"
            #         f" --compute_sampling_error"
            #         f" --sampling_algo ros"
            #         f" --props_learning_rate {plr}"
            #         f" --output_subdir plr_{plr}"
            #     )
            #
            #     mem = 0.3
            #     disk = 0.1
            #     command = f"{mem},{disk},{command}"
            #     print(command)
            #     f.write(command + "\n")


            for plr in [1e-2]:
                command = (
                    f"python ppo_discrete.py"
                    f" --env_id {env_id}"
                    f" --buffer_batches {buffer_batches}"
                    f" --num_steps {num_steps}"
                    f" --learning_rate 0"
                    f" --total_timesteps {total_timesteps}"
                    f" --eval_freq 1"
                    f" --eval_episodes 0"
                    f" --compute_sampling_error"
                    f" --sampling_algo ros"
                    f" --props_learning_rate {plr}"
                    # f" --output_subdir plr_{plr}"
                )

                mem = 0.4
                disk = 0.1
                command = f"{mem},{disk},{command}"
                print(command)
                f.write(command + "\n")