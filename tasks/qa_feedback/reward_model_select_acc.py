import os
import shutil

def select_and_copy_model_dirs(source_folder, destination_folder, desired_model_count):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    model_dirnames = os.listdir(source_folder)
    
    acc_to_dirname = {}
    for dirname in model_dirnames:
        try:
            acc = float(dirname.split('_acc_')[1])
            acc_to_dirname[acc] = dirname
        except (IndexError, ValueError):
            continue  # 如果目录名格式不正确或不存在acc值，则跳过
    
    sorted_accs = sorted(acc_to_dirname.keys())
    
    # 确定最高和最低的acc值(同时去掉特异点)
    min_acc = sorted_accs[1]
    max_acc = sorted_accs[-1]
    
    # 计算差值并根据超参数计算acc_interval
    acc_range = max_acc - min_acc
    acc_interval = acc_range / (desired_model_count - 1)
    
    selected_models = []
    last_acc = min_acc - acc_interval  # 从最小acc开始选取，确保第一个模型被选中
    for acc in sorted_accs:
        if acc - last_acc >= acc_interval or acc == max_acc:  # 确保最高精度的模型被选中
            selected_models.append(acc_to_dirname[acc])
            last_acc = acc
    
    # 复制选中的模型目录
    for model_dir in selected_models:
        src_dir_path = os.path.join(source_folder, model_dir)
        dst_dir_path = os.path.join(destination_folder, model_dir)
        if os.path.isdir(src_dir_path):
            shutil.copytree(src_dir_path, dst_dir_path)
    
    return selected_models, min_acc, max_acc, acc_interval



# 使用示例
# source_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/fact_rm/50epoch_onstep'  # 模型所在的源文件夹路径
# destination_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/fact_rm/selectSet'  # 目标文件夹路径，筛选出的模型将被复制到这里
source_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/rel_rm/50epoch_onstep'  # 模型所在的源文件夹路径
destination_folder = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/rel_rm/selectSet'  # 目标文件夹路径，筛选出的模型将被复制到这里
desired_model_count = 20  # 您想要筛选的模型数量

# 调用函数（请确保路径正确）
selected_models, min_acc, max_acc, acc_interval = select_and_copy_model_dirs(source_folder, destination_folder, desired_model_count)
print("Selected models:", selected_models)
print("Minimum ACC:", min_acc)
print("Maximum ACC:", max_acc)
print("ACC interval:", acc_interval)