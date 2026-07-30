#!/bin/bash

# Test script for DataAnalysisAgent
export PROJECT_ROOT="/Users/z3546829/PycharmProjects/SOCIA"
export DATA_PATH="data_fitting/agent_society/"

echo "Starting DataAnalysisAgent test..."
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATA_PATH: $DATA_PATH"

python main.py \
    --task "Develop a multi-agent simulation system to simulate the review and star a user will comment on a product." \
    --task-file examples/agent_society.json \
    --mode ace \
    --output ./output/agent_society_test_data_analysis \
    --debug

echo "Test completed. Check the output directory for results."
