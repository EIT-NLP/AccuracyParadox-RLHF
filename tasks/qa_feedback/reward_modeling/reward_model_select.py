import os
import shutil
import numpy as np

def parse_metrics(folder_name):
    parts = folder_name.split('_')
    # 初始化一个空字典来存放指标
    metrics = {}
    for i in range(1, len(parts), 2):
        try:
            # 尝试将键值对转换为指标名称和其浮点数值
            key = parts[i]
            value = parts[i + 1]
            # 忽略不是数字的部分
            if value.replace('.', '', 1).isdigit():
                metrics[key] = float(value)
        except ValueError as e:
            # 如果转换失败，输出警告信息
            print(f"Warning: Unable to parse value {value} for key {key} in folder {folder_name}: {e}")
    return metrics


def select_folders(base_dir, metric):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folder_metrics = [(folder, parse_metrics(folder)) for folder in folders]
    folder_metrics = [fm for fm in folder_metrics if fm[1] is not None and metric in fm[1]]  # 确保指标存在

    if not folder_metrics:
        print(f"No valid folders with metric '{metric}' were found.")
        return []

    # 根据指定指标排序
    sorted_folders = sorted(folder_metrics, key=lambda x: x[1][metric], reverse=True)
    
    # 选择acc值最高和最低的两个文件夹
    selected_folders = [sorted_folders[0][0], sorted_folders[-1][0]]
    
    # 计算剩余文件夹的指标差异
    remaining_folders = sorted_folders[1:-1]
    differences = [abs(remaining_folders[i][1][metric] - remaining_folders[i+1][1][metric]) for i in range(len(remaining_folders) - 1)]
    indices = np.argsort(differences)[-3:]  # 获取最大的3个差异的索引
    
    # 根据这些差异选择文件夹
    for i in indices:
        selected_folders.append(remaining_folders[i][0])
    return selected_folders


def copy_folders(folders, src_dir, dest_dir):
    for folder in folders:
        src_folder_path = os.path.join(src_dir, folder)
        dest_folder_path = os.path.join(dest_dir, folder)
        shutil.copytree(src_folder_path, dest_folder_path, dirs_exist_ok=True)  # 允许覆盖已存在的文件夹


def main():
    base_dir = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/baseline_rm'  # Change this to your directory
    metric = 'acc'  # Change to 'f1' if you want to use F1 score

    selected_folders = select_folders(base_dir, metric)
    if not selected_folders:
        print("No folders selected. Exiting.")
        return

    parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
    dest_dir_name = f"{metric.upper()}_select_{os.path.basename(base_dir)}"
    dest_dir = os.path.join(parent_dir, dest_dir_name)

    copy_folders(selected_folders, base_dir, dest_dir)
    print(f"Selected folders copied to {dest_dir}")

if __name__ == "__main__":
    main()
