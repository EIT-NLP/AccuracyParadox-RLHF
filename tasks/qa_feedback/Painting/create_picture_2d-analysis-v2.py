# 导入所需的库
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘制图形
from matplotlib.gridspec import GridSpec  # 导入GridSpec，用于创建复杂的图形布局
import wandb  # 导入Weights & Biases库，用于实验管理
import numpy as np  # 导入NumPy库，用于科学计算
import seaborn as sns  # 导入Seaborn库，用于设置绘图样式
import re  # 导入正则表达式库，用于字符串匹配
import copy

# 设置全局样式
sns.set(style="whitegrid")  # 设置Seaborn的绘图样式为白色网格背景
plt.rcParams['figure.figsize'] = (18, 8)  # 设置全局图形大小为18x8
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 17  # 将图例字体大小设置为18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# 创建Weights & Biases API对象
api = wandb.Api()  # 实例化Weights & Biases的API对象，用于访问实验数据

# 定义要使用的run名称
run_names = [
    # small
    # "2（Test）rel_rm_step_302_f1_0.670_acc_0.670",  # 注释掉的run名称
    # "2（Test）rel_rm_step_602_f1_0.702_acc_0.692",  # 注释掉的run名称
    # "1（Test）fact_rm_step_728_f1_0.647_acc_0.754",  # 选择的run名称
    # "1（Test）fact_rm_step_926_f1_0.661_acc_0.773",  # 选择的run名称
    # "3（Test）comp_rm_step_5130_acc_0.683",  # 注释掉的run名称
    # "3（Test）comp_rm_step_1830_acc_0.700"  # 注释掉的run名称

    # base
    # "2-KL-2500（Test）rel_rm_step_902_f1_0.677_acc_0.686",
    # "2-KL-2900（Test）rel_rm_step_602_f1_0.702_acc_0.692",  # 注释掉的run名称
    # "1（Test）fact_rm_step_2_f1_0.087_acc_0.638",
    # "1（Test）fact_rm_step_926_f1_0.661_acc_0.773",
    # "3（Test）comp_rm_step_2130_acc_0.674",
    # "3（Test）comp_rm_step_1830_acc_0.700",

    # large
    # "2（Test）rel_rm_step_1202_f1_0.682_acc_0.684",
    # "2（Test）rel_rm_step_602_f1_0.702_acc_0.692",
    # "1（Test）fact_rm_step_1256_f1_0.645_acc_0.759",
    # "1（Test）fact_rm_step_926_f1_0.661_acc_0.773",
    "3（Test）comp_rm_step_5730_acc_0.688",
    "3（Test）comp_rm_step_1830_acc_0.700",

]

# 定义要绘制的数据键
keys_to_plot = {
    'reward_raw': "train/train_rm/reward/raw",  # 定义用于绘制的指标键
    'reward_KL': "train/train_rm/reward/KL",  # 定义用于绘制的指标键
    # 'reward_penalized': "train/train_rm/reward/penalized"  # 定义用于绘制的指标键
}

# 根据run数量生成颜色
colors = sns.color_palette("Set2", len(run_names))  # 使用Seaborn的调色板生成颜色

# colors.reverse()  # 反转颜色列表，以便与run名称匹配(仅当相关性任务)

def extract_info_from_name(name):
    """
    从run名称中提取奖励模型的类型和准确度
    """
    if "comp" in name:
        match = re.search(r"acc_(\d+\.\d+)", name)  # 从名称中匹配准确率
        if match:
            acc = match.group(1)
            return f"Acc-{acc}"
        else:
            return name
    else:
        match = re.search(r"f1_(\d+\.\d+)_acc_(\d+\.\d+)", name)  # 从名称中匹配F1得分和准确率
        if match:
            f1_score = match.group(1)
            acc = match.group(2)
            return f"F1-{f1_score}-Acc-{acc}"
        else:
            return name

def replace_labels(labels):
    """
    将列表中的字符串替换为“Most Accurate RM”或“Best Performing RM”
    根据Acc值，较大者替换为“Most Accurate RM”，较小者替换为“Best Performing RM”
    """
    acc_values = []

    # 提取Acc值并存储到acc_values列表中
    for label in labels:
        match = re.search(r"Acc-(\d+\.\d+)", label)
        if match:
            acc_values.append(float(match.group(1)))
        else:
            acc_values.append(None)  # 如果未找到Acc值，则添加None

    # 找到Acc值的最大和最小值及其对应的索引
    if acc_values:
        max_index = acc_values.index(max(filter(None, acc_values)))  # 忽略None值，找到最大值的索引
        min_index = acc_values.index(min(filter(None, acc_values)))  # 忽略None值，找到最小值的索引

        # 将最大值对应的字符串替换为“Most Accurate RM”
        if max_index != -1:
            labels[max_index] = "Most Accurate RM"
        
        # 将最小值对应的字符串替换为“Best Performing RM”
        if min_index != -1:
            labels[min_index] = "Best-Performing RM"

    return labels

def calculate_statistics(data):
    """计算平均值和方差"""
    mean = np.mean(data)  # 计算平均值
    variance = np.var(data)  # 计算方差
    return mean, variance

def plot_performance(metric_key, metric_label, runs, max_steps=None):
    """
    绘制每个run的指标性能图表
    """
    fig = plt.figure(figsize=(18, 8))  # 创建一个图形，大小为18x8
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)  # 使用GridSpec创建一个1x2的网格
    right_gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig, top=0.9, bottom=0.1, left=0.75, right=0.98, hspace=0.5)  # 创建右侧的网格，用于放置两个子图

    # runs.reverse()  # 反转run列表顺序，仅当relevance任务
    

    # 主图
    ax1 = fig.add_subplot(gs[0])  # 在左侧创建主图

    all_values = []  # 创建列表用于存储所有值

    original_labels = {}  # 用于存储每个run的原始label

    for run, color in zip(runs, colors):
        steps = []  # 创建列表用于存储步骤
        values = []  # 创建列表用于存储指标值

        # 获取运行的历史数据
        history = run.history(keys=[metric_key, 'train/step'])

        if not history.empty and metric_key in history and 'train/step' in history:
            step_values = history['train/step'].values  # 获取训练步骤
            metric_values = history[metric_key].values  # 获取指标值

            # 过滤步骤大于max_steps的值
            if max_steps is not None:
                mask = step_values <= max_steps
                step_values = step_values[mask]
                metric_values = metric_values[mask]

            steps.extend(step_values)  # 添加步骤值到列表
            values.extend(metric_values)  # 添加指标值到列表
            all_values.append(metric_values)  # 添加指标值到总列表

        if steps and values:
            # 按步骤排序数据
            sorted_data = sorted(zip(steps, values), key=lambda x: x[0])
            sorted_steps, sorted_values = zip(*sorted_data)

            # 提取信息并修改图例部分
            extracted_info = extract_info_from_name(run.name)
            original_labels[run.name] = extracted_info  # 将原始label存储到字典中

            # 绘制数据点
            ax1.scatter(sorted_steps, sorted_values, label=extracted_info, s=50, color=color)

    # 设置x轴范围
    if max_steps is not None:
        ax1.set_xlim([0, max_steps])

    # 替换labels
    original_labels_list = list(original_labels.values())  # 提取原始labels的值
    replaced_labels = replace_labels(copy.deepcopy(original_labels_list))  # 替换labels

    # 更新图例以使用替换后的labels
    handles, labels = ax1.get_legend_handles_labels()  # 获取当前图例的句柄和标签
    updated_labels = [replaced_labels[original_labels_list.index(label)] for label in labels]  # 根据原始labels更新标签

    ax1.set_xlabel('Training Steps', fontsize=20, labelpad=12)  # 设置x轴标签
    if metric_label == 'reward_raw':
        ax1.set_ylabel('Raw Reward', fontsize=20, labelpad=12)
    elif metric_label == 'reward_KL':
        ax1.set_ylabel('KL Divergence', fontsize=20, labelpad=12)
    elif metric_label == 'reward_penalized':
        ax1.set_ylabel('Penalized Reward', fontsize=20, labelpad=12)# 设置y轴标签
    ax1.grid(True)  # 显示网格
    ax1.legend(handles, updated_labels, loc='upper right')  # 更新图例标签

    # 辅图
    means = []  # 创建列表用于存储均值
    variances = []  # 创建列表用于存储方差
    labels = []  # 创建列表用于存储标签

    for run in runs:
        extracted_info = extract_info_from_name(run.name)  # 提取信息
        labels.append(extracted_info)  # 添加标签到列表

        # 获取历史数据
        history = run.history(keys=[metric_key, 'train/step'])
        if not history.empty and metric_key in history and 'train/step' in history:
            step_values = history['train/step'].values  # 获取训练步骤
            metric_values = history[metric_key].values  # 获取指标值

            # 过滤步骤大于max_steps的值
            if max_steps is not None:
                mask = step_values <= max_steps
                metric_values = metric_values[mask]

            mean, variance = calculate_statistics(metric_values)  # 计算均值和方差
            means.append(mean)  # 添加均值到列表
            variances.append(variance)  # 添加方差到列表

    replace_labels(labels)

    # 平均值图
    ax2 = fig.add_subplot(right_gs[0, 0])  # 创建右上角的子图用于显示均值
    ax2.bar(labels, means, color=colors)  # 绘制条形图
    ax2.set_ylabel('Mean Value', fontsize=20, labelpad=12)  # 设置y轴标签
    ax2.set_xticklabels(labels, fontsize=15, rotation=0, ha='center')  # 设置x轴刻度标签
    ax2.grid(True)  # 显示网格

    # 方差图
    ax3 = fig.add_subplot(right_gs[1, 0])  # 创建右下角的子图用于显示方差
    ax3.bar(labels, variances, color=colors)  # 绘制条形图
    ax3.set_ylabel('Variance', fontsize=20, labelpad=12)  # 设置y轴标签
    ax3.set_xticklabels(labels, fontsize=15, rotation=0, ha='center')  # 设置x轴刻度标签
    ax3.grid(True)  # 显示网格

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/{str(run.name[7:]).split('_')[0]}-{metric_label}.pdf", format='pdf', dpi=1200, bbox_inches='tight')  # 保存图像为高分辨率PDF文件
    plt.show()  # 显示图形

# 获取指定的run，通过查询所有run并匹配名称来找到对应的run对象
# project = "T5-small_RM_research_StepTest_StepOnly"  # 定义项目名称
# project = "T5-base_RM_research_StepTest_StepOnly"  # 定义项目名称
project = "T5-large_RM_research_StepTest_StepOnly"  # 定义项目名称

entity = "battam"  # 定义实体名称

# 使用项目和实体名获取所有run，然后筛选出指定的run
all_runs = api.runs(f"{entity}/{project}")
runs = [run for run in all_runs if any(run_name in run.name for run_name in run_names)]  # 筛选出符合名称的run

# 指定要显示的最大训练步骤，例如2000
max_steps = None
# max_steps = 8800

# 分别绘制每个指标的性能图表
for metric_label, metric_key in keys_to_plot.items():
    plot_performance(metric_key, metric_label, runs, max_steps=max_steps)  # 调用函数绘制性能图表，传入max_steps参数
