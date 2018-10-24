# About
Implementations of deep reinforcement learning algorithms with Tensorflow Eager, such as:

- PPO with CLIP-objective and GAE

# Goals
1. Build intuitions by reimplementing the algorithms.
2. Bring into existence the implementations compatible with Tensorflow's Eager mode and using functionality of the recent versions of Tensorflow instead of obscure implementations branching off older OpenAI code. Specifically:
   - use TF summary writers instead of custom loggers;
   - use TF's distributions instead of custom sampling and custom log-likelihood calculations;
   - use TF datasets instead of custom batching code;
   - use Keras models wherever possible.
3. Build foundation for me to easily experiment with:
   - [MAML](https://arxiv.org/abs/1703.03400)
   - [DeepMimic](https://arxiv.org/abs/1804.02717)-style of reward-shaping using human-provided trajectories

# References
- building on the foundation of:
  - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  - [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
  - [OpenAI Gym](https://github.com/openai/gym)
- inspired by:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)
