MEMDISK = {
    'CartPole-v1': (0.7, 0.3),
    'LunarLander-v2': (0.7, 0.3),
    'Discrete2D100-v0': (0.7, 0.3),
    'Swimmer-v4': (0.7, 0.3),
    'HalfCheetah-v4': (0.7, 0.3),
    'Hopper-v4': (0.7, 0.3),
    'Walker2d-v4': (0.7, 0.3),
    'Ant-v4': (0.7, 0.3),
    'Humanoid-v4': (1, 0.3),
}

TIMESTEPS = {
    'CartPole-v1': int(0.5e6),
    'LunarLander-v2': int(0.5e6),
    'Discrete2D100-v0': int(0.5e6),
    'Swimmer-v4': int(1e6),
    'HalfCheetah-v4': int(1e6),
    'Hopper-v4': int(1e6),
    'Walker2d-v4': int(2e6),
    'Ant-v4': int(2e6),
    'Humanoid-v4': int(4e6),
}

PARAMS = {
    'CartPole-v1': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'LunarLander-v2': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Discrete2D100-v0': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Swimmer-v4': {
        'lr': 1e-3,
        'ns': 8192,
    },
    'HalfCheetah-v4': {
        'lr': 1e-4,
        'ns': 4096,
    },
    'Hopper-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Walker2d-v4': {
        'lr': 1e-4,
        'ns': 2048,
    },
    'Ant-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Humanoid-v4': {
        'lr': 1e-4,
        'ns': 8192,
    },
}


PARAMS = {
    'CartPole-v1': {
        'lr': 1e-4,
        'ns': 256,
    },
    'LunarLander-v2': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Discrete2D100-v0': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Swimmer-v4': {
        'lr': 1e-3,
        'ns': 8192,
    },
    'HalfCheetah-v4': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Hopper-v4': {
        'lr': 1e-3,
        'ns': 4096,
    },
    'Walker2d-v4': {
        'lr': 1e-4,
        'ns': 2048,
    },
    'Ant-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Humanoid-v4': {
        'lr': 1e-4,
        'ns': 8192,
    },
}