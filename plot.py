import os
import numpy as np
from sklearn.metrics import  confusion_matrix, \
    roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import colorsys
import torch

def visualize_grouped_tensors_tsne_3d(tensor_dict, fig_save_path):
    tsne = TSNE(n_components=3, random_state=42, learning_rate='auto')  # 创建T-SNE模型，降维到3维，并显式设置learning_rate
    
    fig = plt.figure(figsize=(10, 6))  # 设置画布大小
    ax = fig.add_subplot(111, projection='3d')  # 创建3D子图
    
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # 定义不同标签的颜色
    for i, (label, tensors) in enumerate(tensor_dict.items()):
        # 将所有张量堆叠成一个矩阵
        tensor_array = np.vstack(tensors)
        
        # 使用T-SNE进行降维
        reduced_data = tsne.fit_transform(tensor_array)
        
        # 绘制降维后的三维点
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                   label=f'Label {label}', color=colors[i % len(colors)], alpha=0.7)
    
    ax.set_title("T-SNE 3D Visualization of Model Outputs")
    ax.set_xlabel("T-SNE Component 1")
    ax.set_ylabel("T-SNE Component 2")
    ax.set_zlabel("T-SNE Component 3")
    ax.legend()
    plt.grid(True)
    plt.savefig(fig_save_path)
    plt.close()


# 可视化函数
def visualize_grouped_tensors(tensor_dict, fig_save_path):
    pca = PCA(n_components=2)  # 创建PCA模型，降维到2维
    
    plt.figure(figsize=(10, 6))  # 设置画布大小
    
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # 定义不同标签的颜色
    for i, (label, tensors) in enumerate(tensor_dict.items()):
        # 将所有张量堆叠成一个矩阵
        tensor_array = np.vstack(tensors)
        
        # 使用PCA进行降维
        reduced_data = pca.fit_transform(tensor_array)
        
        # 绘制降维后的二维点
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                    label=f'Label {label}', color=colors[i % len(colors)], alpha=0.7)
    
    plt.title("PCA Visualization of Model Outputs")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_save_path)
    plt.close()



def calculate_auc(all_outputs, all_labels):
    # 将 all_outputs 列表转换为 (N, 2) 的 numpy 数组
    all_outputs = np.array(all_outputs)
    
    # 将 all_labels 列表转换为 numpy 数组
    all_labels = np.array(all_labels)
    
    # 只使用类别 1 的预测概率 (第二列)
    class_1_scores = all_outputs[:, 1]
    
    # 计算 AUC
    auc = roc_auc_score(all_labels, class_1_scores)
    
    return auc

# def calculate_multiclass_auc(all_outputs, all_labels):
#     # 将A转换为 (N, C) 的 NumPy 数组，B 转换为形状为 (N,)
#     A_array = np.array(all_outputs)  # A: 预测概率数组 (N, C)
#     B_array = np.array(all_labels)  # B: 真实标签 (N,)

#     # 获取类别数量 C
#     N, C = A_array.shape

#     # 将 B 二值化（One-vs-Rest）
#     B_bin = label_binarize(B_array, classes=np.arange(C))  # B_bin 的形状为 (N, C)

#     # 计算每个类别的 AUC
#     aucs = []
#     for i in range(C):
#         # 对于每个类别，计算 AUC
#         auc = roc_auc_score(B_bin[:, i], A_array[:, i])
#         aucs.append(auc)

#     # 计算宏平均 AUC
#     macro_auc = np.mean(aucs)
    
#     return macro_auc


def calculate_multiclass_auc(all_outputs, all_labels):
    # 将A转换为 (N, C) 的 NumPy 数组，B 转换为形状为 (N,)
    A_array = np.array(all_outputs)  # A: 预测概率数组 (N, C)
    B_array = np.array(all_labels)  # B: 真实标签 (N,)

    # 获取类别数量 C
    N, C = A_array.shape

    # 将 B 二值化（One-vs-Rest）
    B_bin = label_binarize(B_array, classes=np.arange(C))  # B_bin 的形状为 (N, C)

    # 计算每个类别的 AUC
    aucs = []
    for i in range(C):
        # 检查当前类别是否只有一种标签
        if len(np.unique(B_bin[:, i])) == 1:
            # 如果只有一种标签，跳过这个类别
            continue
            
        # 对于每个类别，计算 AUC
        try:
            auc = roc_auc_score(B_bin[:, i], A_array[:, i])
            aucs.append(auc)
        except ValueError as e:
            print(f"Warning: Unable to calculate AUC for class {i}: {e}")
            continue

    # 如果没有可以计算AUC的类别，返回None或其他指示值
    if len(aucs) == 0:
        return 0.0
        
    # 计算宏平均 AUC
    macro_auc = np.mean(aucs)
    
    return macro_auc

# def calculate_auc(all_outputs, all_labels):
#     """
#     Calculates AUC score for binary classification.
    
#     Parameters:
#     all_outputs (list of numpy arrays): List of model outputs, each with shape (num_class,)
#     all_labels (list): List of true labels
    
#     Returns:
#     float: AUC score
#     """
#     # Assuming the second class (index 1) is the positive class for AUC calculation
#     positive_class_outputs = np.array([output[1] for output in all_outputs])
#     all_labels = np.array(all_labels)
    
#     # Calculate AUC score
#     auc_score = roc_auc_score(all_labels, positive_class_outputs)
#     return auc_score


def plotCM(all_target, predicted_classes, epoch, fig_save_path):
    cm = confusion_matrix(all_target, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_save_path = os.path.join(fig_save_path, 'cm')
    if not os.path.exists(cm_save_path):
        os.makedirs(cm_save_path)
    plt.savefig(os.path.join(cm_save_path, "epoch_{}_train.png".format(epoch)))
    plt.close()



# def plot_roc_pr_curves(all_outputs, all_labels, epoch, fig_save_path, mode="train"):
#     """
#     Plots ROC and PR curves for each class in a single plot for each type.
    
#     Parameters:
#     all_outputs (list of numpy arrays): List of model outputs, each with shape (num_class,)
#     all_labels (list): List of true labels
#     """
#     num_classes = all_outputs[0].shape[0]
#     all_outputs = np.array(all_outputs)
#     all_labels = np.array(all_labels)
    
#     # Initialize dictionaries to store ROC and PR metrics for each class
#     fpr_dict = {}
#     tpr_dict = {}
#     roc_auc_dict = {}
#     precision_dict = {}
#     recall_dict = {}
#     pr_auc_dict = {}

#     # Calculate ROC and PR metrics for each class
#     for i in range(num_classes):
#         fpr_dict[i], tpr_dict[i], _ = roc_curve(all_labels == i, all_outputs[:, i])
#         roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
#         precision_dict[i], recall_dict[i], _ = precision_recall_curve(all_labels == i, all_outputs[:, i])
#         pr_auc_dict[i] = average_precision_score(all_labels == i, all_outputs[:, i])

#     # Plot all ROC curves in one figure
#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(fpr_dict[i], tpr_dict[i], lw=2, label=f'Class {i} (area = {roc_auc_dict[i]:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.legend(loc="lower right")
#     roc_save_path = os.path.join(fig_save_path, 'roc')
#     if not os.path.exists(roc_save_path):
#         os.makedirs(roc_save_path)
#     plt.savefig(os.path.join(roc_save_path, "epoch_{}_{}.png".format(epoch, mode)))
#     plt.close()

#     # Plot all PR curves in one figure
#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(recall_dict[i], precision_dict[i], lw=2, label=f'Class {i} (area = {pr_auc_dict[i]:.2f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall curve')
#     plt.legend(loc="lower left")
#     pr_save_path = os.path.join(fig_save_path, 'pr')
#     if not os.path.exists(pr_save_path):
#         os.makedirs(pr_save_path)
#     plt.savefig(os.path.join(pr_save_path, "epoch_{}_{}.png".format(epoch, mode)))
#     plt.close()






def plot_roc_pr_curves(all_outputs, all_labels, epoch, fig_save_path, mode="train"):
    """
    绘制 ROC 和 PR 曲线，包含数据验证和异常值处理
    
    Parameters:
    all_outputs (list of numpy arrays): 模型输出概率列表
    all_labels (list): 真实标签列表
    epoch (int): 当前训练轮次
    fig_save_path (str): 图像保存路径
    mode (str): 训练或测试模式
    """
    # 将输入转换为numpy数组并进行数据验证
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # 检查是否存在无效值
    if np.any(np.isnan(all_outputs)) or np.any(np.isinf(all_outputs)):
        print(f"警告: 输出中包含 NaN 或 Inf 值，将进行处理")
        # 将 NaN 和 Inf 替换为 0
        all_outputs = np.nan_to_num(all_outputs, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 确保值在有效范围内
    all_outputs = np.clip(all_outputs, 0, 1)
    
    num_classes = all_outputs.shape[1]
    
    # 初始化存储指标的字典
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    precision_dict = {}
    recall_dict = {}
    pr_auc_dict = {}

    try:
        # 为每个类别计算ROC和PR指标
        for i in range(num_classes):
            try:
                fpr_dict[i], tpr_dict[i], _ = roc_curve(all_labels == i, all_outputs[:, i])
                roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
                
                precision_dict[i], recall_dict[i], _ = precision_recall_curve(all_labels == i, all_outputs[:, i])
                pr_auc_dict[i] = average_precision_score(all_labels == i, all_outputs[:, i])
            except Exception as e:
                print(f"警告: 计算类别 {i} 的指标时出错: {str(e)}")
                continue

        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            if i in roc_auc_dict:
                plt.plot(fpr_dict[i], tpr_dict[i], lw=2, 
                        label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        roc_save_path = os.path.join(fig_save_path, 'roc')
        os.makedirs(roc_save_path, exist_ok=True)
        plt.savefig(os.path.join(roc_save_path, f"epoch_{epoch}_{mode}.png"))
        plt.close()

        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            if i in pr_auc_dict:
                plt.plot(recall_dict[i], precision_dict[i], lw=2,
                        label=f'Class {i} (AP = {pr_auc_dict[i]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        pr_save_path = os.path.join(fig_save_path, 'pr')
        os.makedirs(pr_save_path, exist_ok=True)
        plt.savefig(os.path.join(pr_save_path, f"epoch_{epoch}_{mode}.png"))
        plt.close()

    except Exception as e:
        print(f"错误: 绘制曲线时发生异常: {str(e)}")
        print("输出形状:", all_outputs.shape)
        print("标签形状:", all_labels.shape)
        print("输出范围:", np.min(all_outputs), "-", np.max(all_outputs))


    
    
def analyze_kmer_ra(file_path, label, save_dir = "/xiongjun/test/NEW-MIL/plots" , top_n=10):
    """
    对保存的k-mer和RA文件进行分析，包括RA的分布和RA最大的k-mer的可视化。
    
    :param file_path: k-mer和RA关系文件的路径
    :param top_n: 可视化RA值最大的前n个k-mer，默认为10
    """
    # 读取保存的文件
    df = pd.read_csv(file_path, sep='\t')

    # 检查数据格式
    print("数据概览：")
    print(df.head())

    # 确保数据包含 'k-mer' 和 'Average_RA' 两列
    if 'k-mer' not in df.columns or 'Average_RA' not in df.columns:
        raise ValueError("文件必须包含 'k-mer' 和 'Average_RA' 两列。")

    # 绘制RA的分布图（直方图和密度图）
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Average_RA'].to_numpy(), bins=30, kde=True, color='blue')
    plt.title('RA Distribution')
    plt.xlabel('RA (Relative Abundance)')
    plt.ylabel('Frequency')
    plt.grid(True)
    pic_name = "{}_RA_Distribution".format(label)
    plt.savefig(os.path.join(save_dir, pic_name))
    plt.close()

    # 提取RA值最大的前N个k-mer
    top_kmers = df.nlargest(top_n, 'Average_RA')

    # 输出RA值最大的前N个k-mer
    print(f"\nRA值最大的前{top_n}个k-mer：")
    print(top_kmers)

    # 可视化RA值最大的前N个k-mer（柱状图）
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Average_RA', y='k-mer', data=top_kmers, hue='k-mer', palette='viridis', legend=False)
    plt.title(f'Top {top_n} k-mers with Highest RA')
    plt.xlabel('RA (Relative Abundance)')
    plt.ylabel('k-mer')
    plt.grid(True)
    pic_name = "{}_Top_kmers_with_Highest_RA".format(label)
    plt.savefig(os.path.join(save_dir, '{}_Top_kmers_with_Highest_RA.png'.format(label)))
    plt.close()
    
    
    
    


def plot_purified_labels(Tv, s, Tn, num_class, fig_save_path, label_names_sub=None):
    """
    输入数据（请替换为您的实际数据）
    s = 100  序列数量
    num_class = 4  类别数量
    Tv = torch.softmax(torch.randn(s, num_class), dim=1) 模型估计的概率，形状为 (s, num_class)
    Tn = torch.zeros(s, dtype=torch.long)  原始噪声标签
    """
    # 库标签
    library_label = Tn[0].item()
    # 计算预测标签和预测概率
    predicted_probs, predicted_labels = torch.max(Tv, dim=1)
    predicted_labels = predicted_labels.numpy()
    predicted_probs = predicted_probs.numpy()


    # 分析预测标签
    same_as_library_label = np.sum(predicted_labels == library_label) / s
    zero_label_ratio = np.sum(predicted_labels == 0) / s
    non_zero_label_ratios = [(np.sum(predicted_labels == i) / s) for i in range(1, num_class)]

    print(f"预测标签中与库标签相同的比例: {same_as_library_label:.2f}")
    print(f"预测标签中0标签的占比: {zero_label_ratio:.2f}")
    for i, ratio in enumerate(non_zero_label_ratios, start=1):
        print(f"预测标签中{i}标签的占比: {ratio:.2f}")



    # 决定热图的布局
    n_cols = int(np.floor(np.sqrt(s)))
    n_rows = int(np.ceil(s / n_cols))

    # 创建颜色矩阵
    color_matrix = np.zeros((n_rows, n_cols, 3))  # RGB 颜色

    # 定义用于类别的颜色映射
    cmap = plt.get_cmap('plasma', num_class+1)

    # 填充颜色矩阵
    for idx in range(s):
        row = idx // n_cols
        col = idx % n_cols
        label = predicted_labels[idx]
        prob = predicted_probs[idx]
        # 获取基色（在 RGB 空间）
        base_color = np.array(cmap(label))[:3]
        # 将 RGB 颜色转换为 HLS 颜色
        base_color_hls = colorsys.rgb_to_hls(*base_color)
        # 根据预测概率调整明度（Lightness）
        # 预测概率高时，明度低（颜色深）；预测概率低时，明度高（颜色浅）
        adjusted_lightness = (1 - prob) * 0.5 + 0.1  # 调整明度范围，可以根据需要修改
        adjusted_color_hls = (base_color_hls[0], adjusted_lightness, base_color_hls[2])
        # 将调整后的 HLS 颜色转换回 RGB
        adjusted_color_rgb = colorsys.hls_to_rgb(*adjusted_color_hls)
        color_matrix[row, col] = adjusted_color_rgb

    # 创建图形
    fig, ax = plt.subplots(figsize=(n_cols / 2, n_rows / 2))  # 调整尺寸以适应数据量
    ax.imshow(color_matrix, aspect='equal')

    # 去除坐标轴
    ax.axis('off')

    # 突出显示预测标签与库标签相同的色块
    for idx in range(s):
        if predicted_labels[idx] == library_label:
            row = idx // n_cols
            col = idx % n_cols
            # 在对应的色块上添加边框（例如黑色边框）
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1.5, edgecolor=cmap(num_class+1), facecolor='none')
            ax.add_patch(rect)

    # 添加图例
    from matplotlib.patches import Patch

    # 类别图例
    handles = []
    for i in range(num_class):
        # 获取基色
        base_color = np.array(cmap(i))[:3]
        # 显示颜色比较深的颜色作为图例
        example_lightness = 0.3
        base_color_hls = colorsys.rgb_to_hls(*base_color)
        adjusted_color_hls = (base_color_hls[0], example_lightness, base_color_hls[2])
        adjusted_color_rgb = colorsys.hls_to_rgb(*adjusted_color_hls)
        patch = Patch(facecolor=adjusted_color_rgb, edgecolor='black', label=f'Class {i}' if label_names_sub is None else label_names_sub[i])
        handles.append(patch)

    # 图例位置
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    plt.savefig(fig_save_path)