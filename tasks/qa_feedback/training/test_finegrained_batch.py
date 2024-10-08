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
import subprocess

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

from reward import FineGrainedReward
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
# parser.add_argument("--config", required=False, type=str, help="path to config file", default="/code/FineGrainedRLHF/tasks/qa_feedback/training/fine_grained_config.yml")
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
    
def initialize_models(args, tokenizer, accelerator):
    ref_policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
    
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    
    value = T5Value(
        model_ckpt=args['model']['value_model']['ckpt'],
        model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
        tokenizer=tokenizer,
        accelerator=accelerator,
        freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
    )
    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)
    
    # 指定已训练好的模型路径
    model_file = args['test_model']

    # 现在，根据保存的状态字典加载模型参数
    log_info(f"Evaluating model: {model_file}")
    model_state = torch.load(model_file, map_location=accelerator.device)

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

    return ref_policy, policy, value

def prepare_optimizer_and_scheduler(args, policy, value, accelerator):
    if args['model']['value_model']['policy_value_sharing']:
        parameters = chain(policy.model.parameters(), policy.linear.parameters())
    else:
        parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters(), value.linear.parameters())

    optimizer = torch.optim.Adam(parameters, lr=args['train']['lr'], eps=1e-5)

    total_steps = ceil_div(args['train']['total_episodes'], 
                           args['train']['sampling_batch_size_per_card'] * accelerator.num_processes * args['env']['train_num_samples_per_input'])

    scheduler = transformers.get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=100 * args['train']['n_ppo_epoch_per_rollout'] * accelerator.num_processes,
        num_training_steps=total_steps * args['train']['n_ppo_epoch_per_rollout'] * accelerator.num_processes,
    )

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    return optimizer, scheduler, total_steps

def initialize_reward_models(args, tokenizer, accelerator):
    # 首先定义两个字典，用于存放正常和评估情况下的模型参数
    reward_model_args = {}
    eval_reward_model_args = {}
    
    # 对于每一种模型类型（如relevance, factuality, completeness），提取相应的配置
    for model_type in args['reward']['model_types']:
        # 获取模型配置字典
        model_config = args['reward'].get(f"{model_type}_model", {})
        eval_model_config = args['reward'].get(f"eval_{model_type}_model", {})
        
        # 检查是否需要加载模型，并提取相关配置
        if model_config.get('load_model', False):
            reward_model_args[f"{model_type}_model"] = {
                'ckpt': model_config.get('ckpt'),
                'positive_reward': model_config.get('positive_reward'),
                'negative_reward': model_config.get('negative_reward'),
                'mean': model_config.get('mean', None),
                'std': model_config.get('std', None),
                'bias': model_config.get('bias', None),
                'scale': model_config.get('scale', None),
            }
        
        # 对评估模型进行同样的操作
        if eval_model_config.get('load_model', False):
            eval_reward_model_args[f"{model_type}_model"] = {
                'ckpt': eval_model_config.get('ckpt'),
                'positive_reward': eval_model_config.get('positive_reward'),
                'negative_reward': eval_model_config.get('negative_reward'),
                'mean': eval_model_config.get('mean', None),
                'std': eval_model_config.get('std', None),
                'bias': eval_model_config.get('bias', None),
                'scale': eval_model_config.get('scale', None),
            }
    
    # 初始化奖励模型，传入相应的配置参数
    reward_model = FineGrainedReward(tokenizer=tokenizer, kl_coef=args['ppo']['kl_coef'], sep=args['ppo']['sep'], model_types=args['reward']['model_types'], **reward_model_args)
    eval_reward_model = FineGrainedReward(tokenizer=tokenizer, kl_coef=args['ppo']['kl_coef'], sep=args['ppo']['sep'], model_types=args['reward']['model_types'], **eval_reward_model_args)

    # 准备模型，如果有需要的话
    if getattr(reward_model, 'verbosity_reward', None):
        if hasattr(reward_model.verbosity_reward, 'nf_reward_model') and reward_model.verbosity_reward.nf_reward_model is not None:
            reward_model.verbosity_reward.nf_reward_model = accelerator.prepare(reward_model.verbosity_reward.nf_reward_model)

    if getattr(reward_model, 'factuality_reward', None):
        if hasattr(reward_model.factuality_reward, 'f_reward_model') and reward_model.factuality_reward.f_reward_model is not None:
            reward_model.factuality_reward.f_reward_model = accelerator.prepare(reward_model.factuality_reward.f_reward_model)

    if getattr(reward_model, 'completeness_reward', None):
        if hasattr(reward_model.completeness_reward, 'model') and reward_model.completeness_reward.model is not None:
            reward_model.completeness_reward.model = accelerator.prepare(reward_model.completeness_reward.model)

    if getattr(eval_reward_model, 'verbosity_reward', None):
        if hasattr(eval_reward_model.verbosity_reward, 'nf_reward_model') and eval_reward_model.verbosity_reward.nf_reward_model is not None:
            eval_reward_model.verbosity_reward.nf_reward_model = accelerator.prepare(eval_reward_model.verbosity_reward.nf_reward_model)

    if getattr(eval_reward_model, 'factuality_reward', None):
        if hasattr(eval_reward_model.factuality_reward, 'f_reward_model') and eval_reward_model.factuality_reward.f_reward_model is not None:
            eval_reward_model.factuality_reward.f_reward_model = accelerator.prepare(eval_reward_model.factuality_reward.f_reward_model)

    if getattr(eval_reward_model, 'completeness_reward', None):
        if hasattr(eval_reward_model.completeness_reward, 'model') and eval_reward_model.completeness_reward.model is not None:
            eval_reward_model.completeness_reward.model = accelerator.prepare(eval_reward_model.completeness_reward.model)

    return reward_model, eval_reward_model

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

    # initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                           model_max_length=args['env']['max_input_len'])
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    
    # Load data
    log_info(f'Loading data ...')
    train_dataset = TextGenDataset('train', tokenizer, accelerator=accelerator)
    train_dataloader = DataLoader(train_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                  shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    eval_dataset = TextGenDataset('test', tokenizer, accelerator=accelerator, length_limit=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    ref_policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
    
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    
    value = T5Value(
        model_ckpt=args['model']['value_model']['ckpt'],
        model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
        tokenizer=tokenizer,
        accelerator=accelerator,
        freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
    )
    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)
    
    # # 指定已训练好的模型路径
    # model_file = args['test_model']

    # # 现在，根据保存的状态字典加载模型参数
    # log_info(f"Evaluating model: {model_file}")
    # model_state = torch.load(model_file, map_location=accelerator.device)

    # def adjust_state_dict(state_dict):
    #     """如果需要，添加`module.`前缀，以适配分布式模型结构"""
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # 检查并添加`module.`前缀
    #         new_key = f'module.{k}' if not k.startswith('module.') else k
    #         new_state_dict[new_key] = v
    #     return new_state_dict

    # # 解包模型状态，并根据需要调整
    # policy_model_state_dict = adjust_state_dict(model_state.get('model', {}))
    # policy_linear_state_dict = adjust_state_dict(model_state.get('linear', {}))
    # policy.model.load_state_dict(policy_model_state_dict)
    # policy.linear.load_state_dict(policy_linear_state_dict)

    # if not args['model']['value_model']['policy_value_sharing']:
    #     value_model_state_dict = adjust_state_dict(model_state.get('value_model', {}))
    #     value_linear_state_dict = adjust_state_dict(model_state.get('value_linear', {}))
    #     value.model.load_state_dict(value_model_state_dict)
    #     value.linear.load_state_dict(value_linear_state_dict)

    reward_models, eval_reward_models = initialize_reward_models(args, tokenizer, accelerator)

    # prepare optimizers and schedulers
    optimizer, scheduler, total_steps = prepare_optimizer_and_scheduler(args, policy, value, accelerator)

    # 构建包含奖励模型信息的保存目录
    reward_model_info = {}
    for model_type in args['reward']['model_types']:
        model_key = f"{model_type}_model"
        ckpt = args['reward'][model_key]['ckpt']
        load_model = args['reward'][model_key]['load_model']
        if load_model:
            reward_model_info[model_key] = ckpt.split('/')[-1]

        eval_model_key = f"eval_{model_type}_model"
        eval_ckpt = args['reward'][eval_model_key]['ckpt']
        eval_load_model = args['reward'][eval_model_key]['load_model']
        if eval_load_model:
            reward_model_info[eval_model_key] = eval_ckpt.split('/')[-1]

    # 指定已训练好的模型路径
    model_directory = args["test_model"]
    # 获取目录下所有模型文件
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
    
    # 确保只处理符合预期格式的文件名
    filtered_model_files = [f for f in model_files if f.startswith('model_step_')]

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

        # Set up trainer
        trainer = PPOTrainer(
            args=args,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            ref_policy_model=ref_policy,
            policy_model=policy,
            value_model=value,
            reward_model=reward_models,
            eval_reward_model=eval_reward_models,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            log_info=log_info,
            reward_model_info=reward_model_info,
        )
    
        log_info(f"Starting evaluation for model: {model_path}")
        trainer.valid(step=int(model_file.split('_')[2]))
        log_info(f"Finished evaluation for model: {model_path}")

            
if __name__ == '__main__':
    main()
