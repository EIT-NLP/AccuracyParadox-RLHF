import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import pandas as pd
import wandb
import re

# 设置seaborn样式
sns.set(style="whitegrid")

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 创建wandb API对象
api = wandb.Api()

# 定义项目和实体名称
project = "T5-small_RM_research_StepTest_StepOnly"
# project = "T5-base_RM_research_StepTest_StepOnly"
# project = "T5-large_RM_research_StepTest_StepOnly"
entity = "battam"

# 定义键前缀与实际键的映射关系
keys_prefix_mapping = {
    "1": "eval/eval_rm/factuality_ratios",
    "2": "eval/eval_rm/relevance_ratios",
    "3": "eval/eval_rm/completeness_rewards"
}

def plot_combined_surface_with_xy_projection(x, y, z, title, xlabel, ylabel, zlabel, filename):
    fig = plt.figure(figsize=(16, 8))  # 图片大小设置为适中的长宽比，增强科研美观性

    # 使用GridSpec创建一个1x2的网格
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    # 主图：三维曲面图，放在左侧整个部分
    ax_main = fig.add_subplot(gs[0, 0], projection='3d')
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')

    surface = ax_main.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
    fig.colorbar(surface, ax=ax_main, pad=0.1)
    ax_main.scatter(x, y, z, color='r', s=20, alpha=0.6)

    # 添加等高线
    cset = ax_main.contour(X, Y, Z, zdir='z', offset=min(z), cmap='coolwarm', linestyles='dashed')
    ax_main.clabel(cset, fmt='%2.2f', colors='k', fontsize=10)

    ax_main.set_xlabel(xlabel, fontsize=20, labelpad=12)
    ax_main.set_ylabel(ylabel, fontsize=20, labelpad=12)
    ax_main.set_zlabel(zlabel, fontsize=20, labelpad=12)
    ax_main.view_init(elev=35, azim=120)
    ax_main.invert_yaxis()

    plt.legend(loc='upper left')

    # XY平面投影图，放在右侧部分
    ax_xy = fig.add_subplot(gs[0, 1])
    contour_xy = ax_xy.contourf(X, Y, Z, cmap='viridis', alpha=0.7)
    cbar_xy = fig.colorbar(contour_xy, ax=ax_xy, pad=0.02, label=zlabel)
    cbar_xy.ax.tick_params(labelsize=14)

    ax_xy.set_xlabel(xlabel, fontsize=20, labelpad=12)
    ax_xy.set_ylabel(ylabel, fontsize=20, labelpad=12)
    ax_xy.contour(X, Y, Z, colors='black', linewidths=0.5)  # 添加等高线

    # 保存并展示组合图
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()

# 遍历映射关系中的前缀和键
for prefix, key in keys_prefix_mapping.items():
    runs = api.runs(f"{entity}/{project}")
    accuracies = []
    steps = []
    max_values = []

    for run in runs:
        if run.name.startswith(prefix):
            accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run.name)
            step_match = re.search(r'_step_([0-9]+)', run.name)
            max_step_match = re.search(r'KL-([0-9]+)', run.name)

            if accuracy_match and step_match:
                accuracy = float(accuracy_match.group(1))
                step = int(step_match.group(1))
                history = run.history(keys=[key])

                if max_step_match:
                    max_step = int(max_step_match.group(1)) + 50
                    history = history[history['_step'] <= max_step]

                if not history.empty and key in history:
                    max_value = history[key].max()
                    accuracies.append(accuracy)
                    steps.append(step)
                    max_values.append(max_value)

    if steps and accuracies and max_values and (len(steps) == len(accuracies) == len(max_values)):
        sorted_steps, sorted_accuracies, sorted_max_values = zip(*sorted(zip(steps, accuracies, max_values), key=lambda x: x[0]))

        filename = key.split('/')[1] + "-" + key.split('/')[-1]

        safe_filename = f"/home/llm/FineGrainedRLHF/Pictures/3D-C-v2/3D_Surface-{filename}-{project.split('_')[0]}.pdf"

        plot_combined_surface_with_xy_projection(sorted_steps, sorted_accuracies, sorted_max_values,
                        f'{project.split("_")[0]}-{key.split("/")[-1]}',
                        'RM Trained Steps',
                        'RM Accuracy',
                        'LM Performance',
                        safe_filename)
    else:
        print(f"数据列表长度不一致或存在空列表：{key}")

