# Wireless Networked Control Systems (WNCS) with Deep Reinforcement Learning

This repository implements the paper ["Deep Reinforcement Learning for Wireless Scheduling in Distributed Networked Control"](https://arxiv.org/abs/2109.12562) by Wanchun Liu et al. (2022). It provides a comprehensive framework for studying transmission scheduling in distributed Wireless Networked Control Systems (WNCS) using Deep Reinforcement Learning (DRL).

## Introduction

Wireless Networked Control Systems (WNCS) are a critical technology for Industry 4.0, enabling flexible automation through wireless communication. A key challenge in WNCS is scheduling transmissions between sensors, controllers, and actuators over limited wireless resources. This project implements the fully distributed WNCS model described in the paper, where:

- Multiple plants are controlled over a shared wireless network
- A limited number of frequency channels must be allocated
- Both uplink (sensor→controller) and downlink (controller→actuator) transmissions must be scheduled
- Spatial diversity exists in wireless communication (varying reliability for different links)

The paper presents a novel approach using Deep Reinforcement Learning (DRL) for transmission scheduling to minimize system cost while maintaining stability.

## Repository Structure

```
dopamine/labs/wncs/
│   ├── __init__.py
│   ├── atari_lib.py
│   ├── env.py
│   ├── networks.py
│   ├── run_experiment.py
│   ├── train.py
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   ├── full_rainbow_agent.py
│   │   ├── implicit_quantile_agent.py
│   │   ├── ppo_agent.py
│   │   └── rainbow_agent.py
│   ├── config/
│   │   ├── c51.gin
│   │   ├── dqn.gin
│   │   ├── dqn_eval.gin
│   │   ├── full_rainbow.gin
│   │   ├── implicit_quantile.gin
│   │   ├── ppo.gin
│   │   └── ppo_eval.gin
│   └── environment/
│       ├── __init__.py
│       ├── controller.py
│       ├── environment.py
│       ├── kalman_filter.py
│       └── plant.py
```

## Installation

### Dependencies

This project is built on the [Dopamine RL framework](https://github.com/google/dopamine). To install all required dependencies:

```bash
# Create a virtual environment (recommended)
python -m venv wncs_env
source wncs_env/bin/activate  # On Windows: wncs_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

The `requirements.txt` file should include:
```
tensorflow>=2.4.0
jax>=0.2.9
jaxlib>=0.1.59
flax>=0.3.0
optax>=0.0.6
gin-config>=0.4.0
numpy>=1.19.0
gym>=0.17.0
tqdm>=4.41.0
matplotlib>=3.3.0
```

## Running Experiments

### Quick Start

To run an experiment with default settings (DQN algorithm with 3 plants and 3 frequency channels):

```bash
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/dqn.gin
```

### Selecting Different Algorithms

The repository supports multiple reinforcement learning algorithms. To specify which algorithm to use, select the corresponding gin config file:

```bash
# For DQN
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/dqn.gin

# For PPO
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/ppo.gin

# For Implicit Quantile Networks (IQN)
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/implicit_quantile.gin

# For Full Rainbow
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/full_rainbow.gin
```

### Evaluation Mode

To evaluate a trained agent without further training:

```bash
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/saved/results \
  --gin_files=dopamine/labs/wncs/config/dqn_eval.gin
```

### Custom Configurations

You can override default parameters using the `--gin_bindings` flag:

```bash
python -m dopamine.labs.wncs.train \
  --base_dir=/path/to/save/results \
  --gin_files=dopamine/labs/wncs/config/dqn.gin \
  --gin_bindings="Runner.training_steps=100000" \
  --gin_bindings="Environment.cost_type='stable-cost'"
```

Common customization options include:
- `Environment.include_zeros`: Whether to include idle actions (default: False)
- `Environment.cost_type`: Cost function type ('log-cost', 'stable-cost', or 'state-cost')
- `Environment.aoi_threshold`: Maximum allowed Age-of-Information (default: None)
- `Environment.terminal_cost`: Cost applied when AoI exceeds threshold (default: None)
- `Runner.training_steps`: Number of training steps per iteration
- `Runner.max_steps_per_episode`: Maximum episode length

## Environment Details

### System Model

The environment implements a distributed WNCS with:

- **N plants**: Linear time-invariant systems with process noise
- **M frequencies**: Shared wireless communication channels
- **Smart sensors**: With Kalman filters for state estimation
- **Controller**: Schedules transmissions and generates control commands
- **Actuators**: Apply control commands with buffers for robustness against packet losses

The default configuration uses 3 plants and 3 frequency channels. Each plant is a 2-dimensional LTI system with 2-step controllability, following the paper's specifications in Section 6.

### State Representation

The state is represented by Age-of-Information (AoI) values:
- τ (tau): AoI for sensor measurements at the controller
- η (eta): AoI for control commands at the actuator

### Action Space

Actions determine which of the N plants' uplink or downlink transmissions are scheduled on which of the M frequency channels. The action space size is determined by all valid combinations of assignments.

### Reward/Cost Function

Three cost functions are available:
1. `log-cost`: Logarithmic scaling of empirical costs
2. `stable-cost`: Discretized cost with thresholds (default)
3. `state-cost`: Sum of squared state values

The objective is to minimize the expected total discounted cost, as defined in equation (15) of the paper.

## Agents and Algorithms

The repository implements several reinforcement learning algorithms:

1. **DQN**: Standard Deep Q-Network
2. **Rainbow**: DQN with distributional reinforcement learning
3. **Implicit Quantile Networks (IQN)**: Advanced distributional RL technique
4. **Full Rainbow**: Combines multiple DQN improvements
5. **PPO**: Proximal Policy Optimization (actor-critic method)

The paper specifically uses a Deep Q-learning approach with reduced action space, implemented in `dqn_agent.py`.

## Results and Visualization

Training results are saved in the specified `base_dir`. The system logs:

1. **Average empirical cost**: Main performance metric
2. **Action frequencies**: Distribution of selected actions
3. **Maximum AoI statistics**: Tracking highest τ and η values
4. **Termination statistics**: When episodes end due to AoI violations

Results are automatically saved as numpy files that can be loaded for analysis:
- `log_*.npy`: Average empirical costs
- `action_freq_*.npy`: Action frequency statistics
- `max_aoi_*.npy`: Maximum AoI statistics
- `termination_stats_*.npy`: Termination statistics

To visualize results:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
costs = np.load('path/to/base_dir/log_learning-curve_3-3_dqn_log-cost.npy')

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.xlabel('Episode')
plt.ylabel('Average Empirical Cost')
plt.title('DQN Learning Curve')
plt.grid(True)
plt.savefig('learning_curve.png')
plt.show()
```

## Reproducing Paper Results

To reproduce the main results from the paper:

1. **DQN with reduced action space (Section 5.2)**:
```bash
python -m dopamine.labs.wncs.train \
  --base_dir=results/dqn_reduced \
  --gin_files=dopamine/labs/wncs/config/dqn.gin
```

2. **Comparison with benchmark policies**:
The system automatically logs performance metrics for comparison with the benchmark policies mentioned in the paper (random, round-robin, and greedy policies).

## Troubleshooting

Common issues and solutions:

1. **Out of memory errors**: Reduce batch size or network size in the gin config
```bash
--gin_bindings="ReplayBuffer.batch_size=16"
```

2. **Training instability**: Adjust learning rate
```bash
--gin_bindings="create_optimizer.initial_learning_rate=0.0001"
```

3. **Agent not exploring enough**: Increase exploration parameter
```bash
--gin_bindings="JaxDQNAgent.epsilon_train=0.1"
```

4. **Model not improving**: Try a different cost function
```bash
--gin_bindings="Environment.cost_type='stable-cost'"
```

## Citation

If you use this code, please cite the original paper:

```
@article{liu2022deep,
  title={Deep Reinforcement Learning for Wireless Scheduling in Distributed Networked Control},
  author={Liu, Wanchun and Huang, Kang and Quevedo, Daniel E and Vucetic, Branka and Li, Yonghui},
  journal={arXiv preprint arXiv:2109.12562},
  year={2022}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the [Dopamine RL framework](https://github.com/google/dopamine) developed by Google.
