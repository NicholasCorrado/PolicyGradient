import matplotlib
import seaborn

matplotlib.use('TkAgg')
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_gradient_results(results_path, env_id, output_dir=None):
    """
    Load saved policy gradient results and generate plots with a specific style

    Args:
        results_path: Path to the .npz file containing results
        output_dir: Directory to save the plot (defaults to same directory as results)
    """
    seaborn.set_theme(palette='colorblind', style='whitegrid')
    # Load the results
    results = np.load(results_path)
    print(results['cosine_similarities'])

    # sample_sizes = data['sample_sizes']
    # cosine_similarities = data['cosine_similarities']
    # gradient_magnitudes = data['gradient_magnitudes']
    # l2_norms = data['l2_norms']

    # Extract true gradient norm if it exists
    true_gradient_norm = None
    if 'true_gradient_norm' in results:
        true_gradient_norm = results['true_gradient_norm']

    # Plot results
    plt.figure(figsize=(4 * 3, 4 * 1))

    # Cosine similarity plot
    plt.subplot(1, 3, 1)
    means = [x for x in results['cosine_similarities']]
    stds = [0 for x in results['cosine_similarities']]
    plt.errorbar(results['sample_sizes'], means, yerr=stds, fmt='-o', capsize=5)
    plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Samples')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs Samples Collected')

    # Gradient magnitude plot
    # plt.subplot(1, 3, 2)
    # means = [x[0] for x in results['gradient_magnitudes']]
    # stds = [x[1] for x in results['gradient_magnitudes']]
    # plt.errorbar(results['sample_sizes'], means, yerr=stds, fmt='-o', capsize=5)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.grid(True, alpha=0.3)
    # plt.xlabel('Number of Samples')
    # plt.ylabel('Gradient Magnitude')
    # plt.title('Gradient Magnitude vs Samples Collected')
    #
    # # L2 norm plot
    # plt.subplot(1, 3, 3)
    # means = [x[0] for x in results['l2_norms']]
    # stds = [x[1] for x in results['l2_norms']]
    # plt.errorbar(results['sample_sizes'], means, yerr=stds, fmt='-o', capsize=5)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.grid(True, alpha=0.3)
    # plt.xlabel('Number of Samples')
    # plt.ylabel('L2 Distance')
    # plt.title('L2 Distance vs Samples Collected')

    plt.suptitle(env_id)
    plt.tight_layout()

    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Samples Collected':<12} {'Cosine Similarity':<25} {'Gradient Magnitude':<25} {'L2 Distance':<25}")
    print("-" * 50)

    # for i, n in enumerate(results['sample_sizes']):
    #     cos_mean, cos_std = results['cosine_similarities'][i]
    #     mag_mean, mag_std = results['gradient_magnitudes'][i]
    #     l2_mean, l2_std = results['l2_norms'][i]
    #
    #     print(
    #         f"{n:<12} {cos_mean:.4f} ± {cos_std:.4f} {' ' * 10} {mag_mean:.4f} ± {mag_std:.4f} {' ' * 10} {l2_mean:.4f} ± {l2_std:.4f}")

    # Determine the output path
    # if output_dir is None:
    #     output_dir = os.path.dirname(results_path)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{env_id}.png')

    # Save the plot with high dpi for clarity
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot policy gradient results')
    parser.add_argument('--results-path', type=str, required=False, default='grad_results/GridWorld-5x5-v0/results.npz',
                        help='Path to the .npz file containing results')
    parser.add_argument('--env-id', type=str, default='Discrete2D10-v0')
    parser.add_argument('--output-dir', type=str, default='grad_figures/grad_results',
                        help='Directory to save the plot (defaults to same directory as results)')

    args = parser.parse_args()
    plot_policy_gradient_results(args.results_path, args.env_id, args.output_dir)

'''
[0.10131036 0.24512973 0.36192557 0.6038099 ]
[0.08969704 0.37003818 0.44349498 0.7101601 ]

[0.01008825 0.01978848 0.0330313  0.05428722]
[0.00537554 0.03166069 0.05318051 0.07665935]
[0.2630236  0.35615024 0.4594049  0.6700392 ]

'''