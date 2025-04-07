import os
import glob
import re

# env_id = 'GridWorld-5x5-v0'
#
# for policy_i in range(10, 11):
#     policy_path = f"results/{env_id}/ppo/on_policy/run_1/policy_{policy_i}.pt"
#     output_subdir = f"policy_{policy_i}"
#
#     for seed in range(1,10):
#         command = (
#             f"python grad.py --run_id {seed} --seed {seed}"
#             f" --policy_path {policy_path} "
#             f" --output_rootdir grad_results_tmp"
#             f" --output_subdir {output_subdir} "
#             f" --overwrite_true_grad"
#         )
#         print(command)
#         os.system(command)


env_id = 'GridWorld-5x5-v0'
# env_id = 'HalfCheetah-v4'
for seed in range(100, 200):
    command = (
        f"python grad_sequence.py --run_id {seed} --seed {seed}"
        f" --env_id {env_id}"
        f" --output_rootdir grad_results_tmp3"
    )
    print(command)
    os.system(command)
