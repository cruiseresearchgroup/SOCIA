#!/bin/bash

export PROJECT_ROOT="$(pwd)"
export DATA_PATH="data_fitting/mask_adoption_data/"

python main.py \
    --task "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks." \
    --task-file examples/mask_adoption_task.json \
    --output output/ace_mask_adoption \
    --selfloop 3 \
    --mode ace \
    --auto \
    --iterations 3
