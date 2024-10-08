set -e


# train reward model for COMP
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm-Save.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --validation_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/comp_rm/6k_step_comp_rm \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 12 \
                --per_device_eval_batch_size 12 \
                --eval_steps 30 \
                --evaluation_strategy steps \
                --logging_steps 30 \
                --logging_strategy steps \
                --max_seq_length 2048 \
                --report_to wandb \
                --learning_rate 0.00005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1 \
                --remove_unused_columns False \
                --overwrite_output_dir


# inference for getting mean std of COMP
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm-Save.py \
                --model_name_or_path ./tasks/qa_feedback/model_outputs/comp_rm \
                --validation_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --test_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --output_dir ./tasks/qa_feedback/model_outputs/comp_rm \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 128 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True