CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --multi_gpu \
    tasks/qa_feedback/training/train_finegrained.py --config tasks/qa_feedback/training/fine_grained_config_T5-base.yml
