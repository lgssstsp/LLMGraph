#!/usr/bin/env bash

# # default
# FEATURE_ENGINEERING_LIST=""

# # parse
# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         --feature_engineering_list) FEATURE_ENGINEERING_LIST="$2"; shift ;;
#         *) echo "Unknown parameter passed: $1"; exit 1 ;;
#     esac
#     shift
# done
# # echo "Feature Engineering Techniques: $FEATURE_ENGINEERING_LIST"


# default
FEATURE_ENGINEERING_LIST=""
LEARNING_RATE=""
EPOCH=""

# parse
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --feature_engineering_list)
            FEATURE_ENGINEERING_LIST="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        *) 
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done


# echo "Feature Engineering List: $FEATURE_ENGINEERING_LIST"
# echo "Learning Rate: $LEARNING_RATE"
# echo "Epoch: $EPOCH"


DIR=design
CONFIG=cf
GRID=cf
SAMPLE_ALIAS=cf
REPEAT=5
SAMPLE_NUM=4
MAX_JOBS=1
SLEEP=1
GPU_STRATEGY=greedy

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget

rm -rf results/cf_grid_cf

python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --sample_num $SAMPLE_NUM \
  --out_dir configs

# python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
#   --grid grids/${DIR}/${GRID}.txt \
#   --sample_alias sample/${SAMPLE_ALIAS}.txt \
#   --sample_num $SAMPLE_NUM \
#   --out_dir configs


# #run batch of configs
# #Args: config_dir, num of repeats, max jobs running, sleep time
# bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY
# bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY "$FEATURE_ENGINEERING_LIST"

bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY "$FEATURE_ENGINEERING_LIST" $LEARNING_RATE $EPOCH




# # MAX_JOBS=1
# # bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY

# # GPU_STRATEGY=greedy
# # MAX_JOBS=1
# # bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY

# # MAX_JOBS=1
# # bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $GPU_STRATEGY

# # # aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID} --metric rmse
