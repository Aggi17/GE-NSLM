#!/usr/bin/env bash

# Training script for RoBERTa-based fake news detection model
# Usage: sh train_roberta.sh [lambda] [prior] [mask]

MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=roberta-large    
VERSION=v5
MAX_NUM_QUESTIONS=8
DATASET="fakeddit"

# Sequence length configurations
MAX_SEQ1_LENGTH=100  # Claim text length
MAX_SEQ2_LENGTH=12   # Question text length
MAX_SEQ3_LENGTH=45   # Image caption length
CAND_K=3

# Training parameters
LAMBDA=${1:-0.5}     # Logic lambda parameter
PRIOR=${2:-random}   # Prior type: uniform/nli/random
MASK=${3:-0.0}       # Mask rate
echo "lambda = $LAMBDA, prior = $PRIOR, mask = $MASK"

# Path configurations - replace with your actual paths
DATA_DIR=$PJ_HOME/data/${DATASET}/
OUTPUT_DIR=$PJ_HOME/output/${DATASET}/${DATASET}_${PRIOR}_l${LAMBDA}

# Training hyperparameters
NUM_TRAIN_EPOCH=7
GRADIENT_ACCUMULATION_STEPS=2
PER_GPU_TRAIN_BATCH_SIZE=8 
PER_GPU_EVAL_BATCH_SIZE=16

LOGGING_STEPS=400 
SAVE_STEPS=400  

# Model architecture parameters
HS=128 
SHARE_HS=128

# Run training
python3 train.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_seq1_length ${MAX_SEQ1_LENGTH} \
  --max_seq2_length ${MAX_SEQ2_LENGTH} \
  --max_seq3_length ${MAX_SEQ3_LENGTH} \
  --max_num_questions ${MAX_NUM_QUESTIONS} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --num_train_epochs ${NUM_TRAIN_EPOCH} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
  --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
  --logging_steps ${LOGGING_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --cand_k ${CAND_K} \
  --logic_lambda ${LAMBDA} \
  --prior ${PRIOR} \
  --overwrite_output_dir \
  --temperature 1.0 \
  --hs ${HS} \
  --share_hs ${SHARE_HS}

# Send completion notification
python3 cjjpy.py --lark "$OUTPUT_DIR fact checking training completed"