import numpy as np

from dopamine.labs.wncs.environment.environment import Environment
from dopamine.labs.wncs.environment.plant import Plant

import gin

@gin.configurable
def create_wncs_environment():
    # define Environment related parameters (controllability, no of used frequencies and plants)
    controllability = 2
    no_of_channels = 3

    plants = [
        Plant(2,
                controllability,
                np.array([[1.1, 0.2], [0.2, 0.8]]),
                np.array([[1], [1]]),
                np.identity(2),
                0.1 * np.identity(2),
                0.1 * np.identity(2),
                np.array([[-2.900, 1.000]])),
        Plant(2,
                controllability,
                np.array([[1.2, 0.2], [0.2, 0.9]]),
                np.array([[1], [1]]),
                np.identity(2),
                0.1 * np.identity(2),
                0.1 * np.identity(2),
                np.array([[-3.533, 1.433]])),
        Plant(2,
                controllability,
                np.array([[1.2, 0.2], [0.2, 0.9]]),
                np.array([[1], [1]]),
                np.identity(2),
                0.1 * np.identity(2),
                0.1 * np.identity(2),
                np.array([[-3.533, 1.433]])),
        Plant(2,
                controllability,
                np.array([[1.3, 0.2], [0.2, 1.0]]),
                np.array([[1], [1]]),
                np.identity(2),
                0.1 * np.identity(2),
                0.1 * np.identity(2),
                np.array([[-4.233, 1.933]])),
        Plant(2,
                controllability,
                np.array([[1.3, 0.2], [0.2, 1.0]]),
                np.array([[1], [1]]),
                np.identity(2),
                0.1 * np.identity(2),
                0.1 * np.identity(2),
                np.array([[-4.233, 1.933]]))
    ]

#     plants = [
#         Plant(2,
#               controllability,
#               np.array([[1.1, 0.2], [0.2, 0.8]]),
#               np.array([[1], [1]]),
#               np.identity(2),
#               0.1 * np.identity(2),
#               0.1 * np.identity(2),
#               np.array([[-2.900, 1.000]])
#               ),
#         Plant(2,
#               controllability,
#               np.array([[1.2, 0.2], [0.2, 0.9]]),
#               np.array([[1], [1]]),
#               np.identity(2),
#               0.1 * np.identity(2),
#               0.1 * np.identity(2),
#               np.array([[-3.533, 1.433]])),
#         Plant(2,
#               controllability,
#               np.array([[1.3, 0.2], [0.2, 1.0]]),
#               np.array([[1], [1]]),
#               np.identity(2),
#               0.1 * np.identity(2),
#               0.1 * np.identity(2),
#               np.array([[-4.233, 1.933]]))
#     ]

    # Generate random success rates for uplink and downlink channels, fix the random seed to check the results later.
    np.random.seed(112)
    uplink_coefficients = np.random.uniform(0.65, 1, (no_of_channels, len(plants)))
    downlink_coefficients = np.random.uniform(0.65, 1, (no_of_channels, len(plants)))
    env = Environment(plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability)
    
    return env