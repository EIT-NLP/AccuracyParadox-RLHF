import os
import shutil

def select_and_copy_model_dirs_by_step(source_folder, destination_folder, desired_model_count):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    model_dirnames = os.listdir(source_folder)
    
    step_to_dirname = {}
    for dirname in model_dirnames:
        parts = dirname.split('_')
        try:
            # 寻找包含'step'文本的部分，并提取其后的数字作为步骤值
            step_index = parts.index('step') + 1
            step = int(parts[step_index])
            step_to_dirname[step] = dirname
        except (ValueError, IndexError):
            continue  # 如果目录名格式不正确或不存在step值，则跳过
    
    sorted_steps = sorted(step_to_dirname.keys())
    
    # 根据步骤值均匀选取模型
    step_interval = max(1, len(sorted_steps) // desired_model_count)
    selected_models = [step_to_dirname[step] for i, step in enumerate(sorted_steps) if i % step_interval == 0]
    
    # 如果选取的模型少于desired_model_count，尝试从剩余模型中补充
    additional_models_needed = desired_model_count - len(selected_models)
    if additional_models_needed > 0:
        additional_models = [step_to_dirname[step] for i, step in enumerate(sorted_steps) if i % step_interval != 0][:additional_models_needed]
        selected_models.extend(additional_models)
    
    # 确保选中模型的数量不超过desired_model_count
    selected_models = selected_models[:desired_model_count]
    
    # 复制选中的模型目录
    for model_dir in selected_models:
        src_dir_path = os.path.join(source_folder, model_dir)
        dst_dir_path = os.path.join(destination_folder, model_dir)
        if os.path.isdir(src_dir_path):
            shutil.copytree(src_dir_path, dst_dir_path)
    
    return selected_models

# 使用示例
# source_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/rel_rm/50epoch_onstep2'  # 模型所在的源文件夹路径
# destination_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/rel_rm/selectSet_step'  # 目标文件夹路径，筛选出的模型将被复制到这里

# source_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/fact_rm/50epoch_onstep2'  # 模型所在的源文件夹路径
# destination_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/fact_rm/selectSet_step'  # 目标文件夹路径，筛选出的模型将被复制到这里

source_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/comp_rm/6k_step_comp_rm'  # 模型所在的源文件夹路径
destination_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/comp_rm/selectSet_step'  # 目标文件夹路径，筛选出的模型将被复制到这里

desired_model_count = 20  # 您想要筛选的模型数量

selected_models = select_and_copy_model_dirs_by_step(source_folder, destination_folder, desired_model_count)
print("Selected models:", selected_models)
