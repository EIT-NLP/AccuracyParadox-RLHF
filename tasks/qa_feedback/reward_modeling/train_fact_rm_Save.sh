set -e

# train reward model for F-ERR_sentence
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_fg_rm-Save.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/F-ERR_sentence/train.json \
                --validation_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --test_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/fact_rm/50epoch_onstep2 \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 24 \
                --per_device_eval_batch_size 24 \
                --evaluation_strategy steps \
                --eval_steps 2 \
                --logging_strategy steps \
                --save_strategy no \
                --max_seq_length 2048 \
                --report_to wandb \
                --learning_rate 0.000005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1 \
                --overwrite_output_dir