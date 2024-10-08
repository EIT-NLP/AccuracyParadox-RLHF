set -e

# Evaluate reward model for F-ERR_sentence
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/eval_fg_rm.py \
                --model_name_or_path ./tasks/qa_feedback/model_outputs/fact_rm/50epoch_test/model_epoch_17.0_f1_0.664_acc_0.770 \
                --validation_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/fact_rm/evaluation \
                --do_eval \
                --bf16 \
                --per_device_eval_batch_size 24 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --max_seq_length 2048 \
                --report_to wandb \

