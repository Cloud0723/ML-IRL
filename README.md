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
- We use yaml files in `configs/` for experimental configurations, please change `obj` value (in the first line) for each method, here is the list of `obj` values:
    -  Our methods (ML-IRL): ML_S: `maxentirl`, ML_SA: `maxentirl_sa`
- After running, you will see the training logs in `logs/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Sec 1 IRL benchmark (MuJoCo)
First, make sure that you have downloaded expert data into `expert_data/`. *Otherwise*, you can generate expert data by training expert policy:
```bash
python common/train_gd.py configs/samples/experts/{env}.yml # env is in {hopper, walker2d, halfcheetah, ant}
python common/collect.py configs/samples/experts/{env}.yml # env is in {hopper, walker2d, halfcheetah, ant}
```

Then train our method with provided expert data method (Policy Performance).

```bash
# you can vary obj in {`maxentirl_sa`, `maxentirl`}
python ml/irl_samples.py configs/samples/agents/{env}.yml
```

### Sec 2 Transfer task
First, make sure that you have downloaded expert data into `expert_data/`. *Otherwise*, you can generate expert data by training expert policy:
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
