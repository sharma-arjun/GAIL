# PyTorch implementation of GAE and GAIL with PPO

This repository contains a Pytorch implementation of Generalized Advantage Estimation (GAE) and Generative Adversarial Imitation Learning (GAIL) with Proximal Policy Optimization (PPO)

## Usage

For GAE, use

```
python gae.py --env-name Hopper-v1
```

For GAIL, use

```
python gail.py --env-name Hopper-v1 --expert-path hopper_expert_trajectories/
```

For GAIL with Phase MLP architecture, use

```
python phase_gail.py --env-name Hopper-v1 --expert-path hopper_expert_trajectories/
```
