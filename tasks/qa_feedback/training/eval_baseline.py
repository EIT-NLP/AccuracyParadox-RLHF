import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb
import yaml
import nltk

from fgrlhf.ppo import PPOTrainer
from fgrlhf.policy import T5Policy
from fgrlhf.value import T5Value
from fgrlhf.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

from reward import BaselineReward
from collections import OrderedDict

logging.basicConfig(level=logging.ERROR)

# prepare accelerator and logger
accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)
        
# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="path to config file")
args = parser.parse_args()
# load yaml file
with open(args.config) as f:
    args =yaml.safe_load(f)


# prepare data
class TextGenDataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None):
        super().__init__()
        
        self.split = split
        self.dataset_fns = {
            "train": "tasks/qa_feedback/data/train.json",
            "dev": "tasks/qa_feedback/data/dev.json",
            "test": "tasks/qa_feedback/data/test.json"
        }
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:
            instances.append({
                "prompt": task_instance['text'],
                "metadata": {
                    "prompt": task_instance['text'],
                    "references": task_instance['answer'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],}
            })
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata
        }
        return result
    

def main():
    # set seed
    set_seed(args['train']['seed'], args['train']['cuda_deterministic'])

    # set saving directories
    log_info(f"Write to output directory: {args['logging']['save_dir']}")
    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        # save the config file
        with open(os.path.join(args['logging']['save_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    # initialize policy and value model tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                           model_max_length=args['env']['max_input_len'])
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    
    # Load data for evaluation
    log_info(f'Loading data for evaluation...')
    eval_dataset = TextGenDataset('dev', tokenizer, accelerator=accelerator, length_limit=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models
    log_info(f'Initializing models for evaluation...')
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    value = T5Value(
        model_ckpt=args['model']['value_model']['ckpt'],
        model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
        tokenizer=tokenizer,
        accelerator=accelerator,
        freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
    )

    # Prepare models for training or evaluation
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)

    # Initialize reward model for evaluation
    eval_reward_model = BaselineReward(
        tokenizer=tokenizer,
        baseline_model_ckpt=args['reward']['eval_baseline_model']['ckpt'],
        kl_coef=args['ppo']['kl_coef'],
        baseline_reward_mean=args['reward']['eval_baseline_model']['mean'],
        baseline_reward_std=args['reward']['eval_baseline_model']['std'],
        baseline_reward_bias=args['reward']['eval_baseline_model']['bias'],
        baseline_reward_scale=args['reward']['eval_baseline_model']['scale'],
    )
    eval_reward_model.baseline_reward.model = accelerator.prepare(eval_reward_model.baseline_reward.model)

    # 指定已训练好的模型路径
    model_directory = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/baseline'
    # 获取目录下所有模型文件
    model_files = [f for f in os.listdir(model_directory) if f.endswith('_0.510.pth')]
    
    # 确保只处理符合预期格式的文件名
    filtered_model_files = [f for f in model_files if f.startswith('model_step_') and '_baseline_reward_model_' in f]

    # 对模型文件按照训练步骤进行排序
    sorted_model_files = sorted(filtered_model_files, key=lambda x: int(x.split('_')[2]))

    # 现在，根据保存的状态字典加载模型参数
    for model_file in sorted_model_files:
        model_path = os.path.join(model_directory, model_file)
        log_info(f"Evaluating model: {model_path}")
        model_state = torch.load(model_path, map_location=accelerator.device)

        def adjust_state_dict(state_dict):
            """如果需要，添加`module.`前缀，以适配分布式模型结构"""
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 检查并添加`module.`前缀
                new_key = f'module.{k}' if not k.startswith('module.') else k
                new_state_dict[new_key] = v
            return new_state_dict

        # 解包模型状态，并根据需要调整
        policy_model_state_dict = adjust_state_dict(model_state.get('model', {}))
        policy_linear_state_dict = adjust_state_dict(model_state.get('linear', {}))
        policy.model.load_state_dict(policy_model_state_dict)
        policy.linear.load_state_dict(policy_linear_state_dict)

        if not args['model']['value_model']['policy_value_sharing']:
            value_model_state_dict = adjust_state_dict(model_state.get('value_model', {}))
            value_linear_state_dict = adjust_state_dict(model_state.get('value_linear', {}))
            value.model.load_state_dict(value_model_state_dict)
            value.linear.load_state_dict(value_linear_state_dict)


        # Create and use PPOTrainer instance for evaluation with loaded models
        trainer = PPOTrainer(
            args=args,
            train_dataloader=None,  # Training dataset not needed
            eval_dataloader=eval_dataloader,  # Evaluation dataset
            ref_policy_model=None,  # Reference policy model not needed
            policy_model=policy,
            value_model=value,
            reward_model=None,  # Training reward model not needed
            optimizer=None,  # Optimizer not needed
            scheduler=None,  # Scheduler not needed
            accelerator=accelerator,
            log_info=log_info,
            eval_reward_model=eval_reward_model,  # Evaluation reward model
        )
        log_info(f"Starting evaluation for model: {model_path}")
        trainer.valid(step=int(model_file.split('_')[2]))
        log_info(f"Finished evaluation for model: {model_path}")

if __name__ == '__main__':
    main()
