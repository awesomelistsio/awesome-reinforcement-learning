# Awesome Reinforcement Learning [![Awesome Lists](https://srv-cdn.himpfen.io/badges/awesome-lists/awesomelists-flat.svg)](https://github.com/awesomelistsio/awesome)

[![GitHub Sponsors](https://srv-cdn.himpfen.io/badges/github/github-flat.svg)](https://github.com/sponsors/awesomelistsio) &nbsp; 
[![Ko-Fi](https://srv-cdn.himpfen.io/badges/kofi/kofi-flat.svg)](https://ko-fi.com/awesomelists) &nbsp; 
[![PayPal](https://srv-cdn.himpfen.io/badges/paypal/paypal-flat.svg)](https://www.paypal.com/donate/?hosted_button_id=3LLKRXJU44EJJ) &nbsp; 
[![Stripe](https://srv-cdn.himpfen.io/badges/stripe/stripe-flat.svg)](https://tinyurl.com/e8ymxdw3) &nbsp; 
[![X](https://srv-cdn.himpfen.io/badges/twitter/twitter-flat.svg)](https://x.com/ListsAwesome) &nbsp; 
[![Facebook](https://srv-cdn.himpfen.io/badges/facebook-pages/facebook-pages-flat.svg)](https://www.facebook.com/awesomelists)

> A curated list of awesome frameworks, libraries, tools, environments, tutorials, research papers, and resources for reinforcement learning (RL). This list covers fundamental concepts, advanced algorithms, applications, and popular frameworks for building RL models.

## Contents

- [Frameworks and Libraries](#frameworks-and-libraries)
- [Tools and Environments](#tools-and-environments)
- [Core Algorithms](#core-algorithms)
- [Advanced Algorithms](#advanced-algorithms)
- [Applications](#applications)
- [Learning Resources](#learning-resources)
- [Research Papers](#research-papers)
- [Books](#books)
- [Community](#community)
- [Contribute](#contribute)
- [License](#license)

## Frameworks and Libraries

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - A reliable set of implementations of reinforcement learning algorithms in Python.
- [Ray RLlib](https://docs.ray.io/en/latest/rllib.html) - A scalable reinforcement learning library built on top of Ray.
- [TF-Agents](https://www.tensorflow.org/agents) - A library for reinforcement learning using TensorFlow.
- [OpenAI Baselines](https://github.com/openai/baselines) - A collection of high-quality implementations of RL algorithms by OpenAI.
- [Dopamine](https://github.com/google/dopamine) - A research framework by Google focused on fast prototyping of RL algorithms.
- [Acme](https://github.com/deepmind/acme) - A library by DeepMind for building and testing reinforcement learning agents.

## Tools and Environments

- [OpenAI Gym](https://www.gymlibrary.dev/) - A toolkit for developing and comparing RL algorithms with a variety of environments.
- [DeepMind Control Suite](https://www.deepmind.com/research/open-source/deepmind-control-suite) - A set of Python-based reinforcement learning environments.
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - A toolkit by Unity for training intelligent agents using RL.
- [PyBullet](https://pybullet.org/wordpress/) - An open-source Python module for physics simulations in RL.
- [PettingZoo](https://www.pettingzoo.ml/) - A library of multi-agent reinforcement learning environments.
- [CARLA Simulator](https://carla.org/) - An open-source simulator for autonomous driving research using RL.

## Core Algorithms

- [Deep Q-Learning (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) - A value-based method using deep learning to approximate the Q-value function.
- [Policy Gradient Methods](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) - A class of algorithms that directly optimize the policy.
- [Actor-Critic Methods](https://arxiv.org/abs/1602.01783) - Algorithms that use both policy (actor) and value (critic) functions.
- [REINFORCE Algorithm](https://www.cs.cmu.edu/~sutton/book/ebook/node65.html) - A Monte Carlo policy gradient method for training RL agents.
- [SARSA (State-Action-Reward-State-Action)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) - An on-policy RL algorithm.

## Advanced Algorithms

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - A stable and efficient policy optimization method.
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) - An off-policy algorithm for continuous action spaces.
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) - An entropy-regularized algorithm for stable learning in continuous action spaces.
- [A3C (Asynchronous Advantage Actor-Critic)](https://arxiv.org/abs/1602.01783) - An efficient, asynchronous RL algorithm for training agents.
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) - An algorithm designed to maintain stable updates of the policy.

## Applications

- **Autonomous Vehicles**: Using RL to train self-driving cars in simulators like CARLA and AirSim.
- **Game AI**: RL is widely used for creating intelligent agents in games like chess, Go, and video games (e.g., OpenAI’s Dota 2).
- **Robotics**: Training robots to perform tasks using RL environments like MuJoCo and PyBullet.
- **Financial Trading**: Using RL for algorithmic trading and portfolio optimization.
- **Healthcare**: Applying RL for personalized treatment strategies and drug discovery.

## Learning Resources

- [Coursera: Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) - A series of courses on RL by the University of Alberta.
- [Deep Reinforcement Learning Nanodegree (Udacity)](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) - A program focused on deep RL techniques.
- [DeepMind’s RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series) - A comprehensive RL course by DeepMind researchers.
- [David Silver’s Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) - A popular course by David Silver on RL concepts and algorithms.

## Research Papers

- [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602) - The seminal paper introducing DQN.
- [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347) - The paper introducing PPO, a popular policy gradient method.
- [Asynchronous Methods for Deep Reinforcement Learning (2016)](https://arxiv.org/abs/1602.01783) - The introduction of A3C, a highly efficient RL algorithm.
- [Deep Reinforcement Learning with Double Q-learning (2016)](https://arxiv.org/abs/1509.06461) - A paper that addresses the overestimation bias of Q-learning.
- [Curiosity-driven Exploration by Self-supervised Prediction (2017)](https://arxiv.org/abs/1705.05363) - A method for encouraging exploration in RL agents.

## Books

- *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto - The classic textbook on RL.
- *Deep Reinforcement Learning Hands-On* by Maxim Lapan - A practical guide to RL with PyTorch.
- *Algorithms for Reinforcement Learning* by Csaba Szepesvári - A comprehensive introduction to RL algorithms.
- *Hands-On Reinforcement Learning with Python* by Sudharsan Ravichandiran - A book covering practical RL implementations in Python.

## Community

- [Reddit: r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) - A subreddit dedicated to discussions on RL research and applications.
- [OpenAI Forum](https://community.openai.com/) - A place to discuss OpenAI’s RL research and projects.
- [DeepMind Blog](https://www.deepmind.com/blog) - A blog covering DeepMind’s latest research in RL.
- [Discord: Reinforcement Learning Community](https://discord.com/invite/rl) - A Discord server for discussing RL topics.
- [RLlib Users Group](https://groups.google.com/g/rllib-users) - A forum for discussing Ray’s RLlib.

## Contribute

Contributions are welcome!

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)](http://creativecommons.org/licenses/by-sa/4.0/)
