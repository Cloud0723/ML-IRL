# Maximum-Likelihood Inverse Reinforcement Learning with Finite-Time Guarantees
ML-IRL is an algorithm for inverse reinforcement learning that is discussed in the Neurips paper [link](https://proceedings.neurips.cc/paper_files/paper/2022/file/41bd71e7bf7f9fe68f1c936940fd06bd-Paper-Conference.pdf) and AISTAT paper [link](https://openreview.net/forum?id=j4CbQGb0iF&referrer=%5Bthe%20profile%20of%20Chenliang%20Li%5D(%2Fprofile%3Fid%3D~Chenliang_Li3))

You can download our expert data from the [google_drive](https://drive.google.com/drive/folders/1qD8mLj4MDuH5TFR6c2fwNrfuZF3RR5QA?usp=drive_link)
## Installation
- PyTorch 1.5+
- OpenAI Gym
- [MuJoCo](https://www.roboti.us/license.html)
- `pip install ruamel.yaml` 


## File Structure
- ML-IRL (our method): `ml/`
- SAC agent: `common/`
- Environments: `envs/`
- Configurations: `configs/`

## Instructions
- All the experiments are to be run under the root folder. 
- Before starting experiments, please `export PYTHONPATH=${PWD}:$PYTHONPATH` for env variable. 
- We use yaml files in `configs/` for experimental configurations, Please change `obj` value (in the first line) for each method, here is the list of `obj` values:
    -  Our methods (ML-IRL): ML_S: `maxentirl`, ML_SA: `maxentirl_sa`
- After running, you will see the training logs in `logs/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Sec 1 IRL benchmark (MuJoCo)
First, you can generate expert data by training expert policy:
```bash
python common/train_gd.py configs/samples/experts/{env}.yml # env is in {hopper, walker2d, halfcheetah, ant}
python common/collect.py configs/samples/experts/{env}.yml # env is in {hopper, walker2d, halfcheetah, ant}
```

Then train our method with the provided expert data method (Policy Performance).

```bash
# you can vary obj in {`maxentirl_sa`, `maxentirl`}
python ml/irl_samples.py configs/samples/agents/{env}.yml
```

### Sec 2 Transfer task
First, you can generate expert data by training expert policy.
Make sure that the `env_name` parameter in `configs/samples/experts/ant_transfer.yml` is set to `CustomAnt-v0`
```bash
python common/train_gd.py configs/samples/experts/ant_transfer.yml
python common/collect.py configs/samples/experts/ant_transfer.yml
```

After the training is done, you can choose one of the saved reward model to train a policy from scratch (Recovering the Stationary Reward Function).

Transferring the reward to disabled Ant

```bash 
python common/train_optimal.py configs/samples/experts/ant_transfer.yml
python ml/irl_samples.py configs/samples/agents/data_transfer.yml(data transfer)
```
