#!/bin/bash
# python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts=1
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python run_expert.py experts/$e.pkl $e --render --num_rollouts=1
done
