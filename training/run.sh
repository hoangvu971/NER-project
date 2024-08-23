#!/bin/bash
python fine_tune.py \
  --model_name_or_path ../models/gliner_medium-v2.1 \
  --data_path ../data/synthetic-pii-ner.json \
  --output_dir ../models/gliner_medium-v2.1 \
  --learning_rate 5e-6 \
  --weight_decay 0.01 \
  --others_lr 1e-5 \
  --others_weight_decay 0.01 \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --evaluation_strategy "steps" \
  --save_total_limit 10 \
  --dataloader_num_workers 0 \
  --use_cpu False \
  --report_to "mlflow"