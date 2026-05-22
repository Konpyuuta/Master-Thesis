model_name=UniTS_zeroshot
exp_name=UniTS_zeroshot_pretrain_x64
wandb_mode=offline
ptune_name=zeroshot_newdata

d_model=128

random_port=$((RANDOM % 9000 + 1000))



# Zero-shot test on new forecasting datasets
# Note: The inference in this code test all samples of the dataset, 
# which is not the same as the original paper that only test 1 sample for each dataset.
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 0 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --batch_size 1 \
  --task_name 'imputation' \
  --subsample_pct 0.001 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --pretrained_weight units_x128_pretrain_checkpoint.pth \
  --task_data_config_path  data_provider/imputation.yaml