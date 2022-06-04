export PYTHONPATH=${PWD}:$PYTHONPATH
# UNIT TEST

#MUJOCO DATA COLLECT
#python common/train_gd.py configs/samples/experts/gd.yml
#python common/collect.py configs/samples/experts/gd.
#python common/train_gd.py configs/samples/experts/hopper.yml
#python common/train_gd.py configs/samples/experts/halfcheetah.yml
#python common/train_gd.py configs/samples/experts/walker2d.yml
#python common/collect.py configs/samples/experts/humanoid.yml

# MUJOCO IRL BENCHMARK
python ml/irl_samples.py configs/samples/agents/ant.yml

#MUJOCO TRANSFER
#python common/train_optimal.py configs/samples/experts/ant_transfer.yml
#python ml/irl_samples.py configs/samples/agents/data_transfer.yml(data transfer)