# 此处是神经网络训练与评估代码
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import psutil  # 用于 CPU 和内存监控
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, \
    roc_curve, precision_recall_curve
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# 可选：Windows 下监控 NVIDIA GPU
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
except Exception as e:
    print(f"[WARN] NVML 初始化失败: {e}")
    _NVML_AVAILABLE = False


def get_hardware_stats(device):
    stats = {}

    # CPU 和内存（跨平台）
    stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    stats['ram_used_gb'] = mem.used / (1024 ** 3)
    stats['ram_total_gb'] = mem.total / (1024 ** 3)

    # MPS (Mac) 显存监控
    if device.type == 'mps':
        stats['gpu_allocated_mb'] = torch.mps.current_allocated_memory() / (1024 ** 2)
        stats['gpu_driver_mb'] = torch.mps.driver_allocated_memory() / (1024 ** 2)

    # CUDA (Windows/Linux) 显存监控
    elif device.type == 'cuda' and _NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认用第一个被识别到的 NVIDIA GPU
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats['gpu_util_percent'] = util.gpu
            stats['gpu_mem_used_mb'] = mem_info.used / (1024 ** 2)
            stats['gpu_mem_total_mb'] = mem_info.total / (1024 ** 2)
        except:
            pass

    return stats


class G2D2Dataset(Dataset):
    def __init__(self, pairs, labels, drug_to_genes, disease_to_genes, gene_to_idx, gene_dim):
        self.gene_dim = gene_dim
        self.labels = np.array(labels, dtype=np.float32)

        drug_gene_lists = [drug_to_genes.get(p[0], []) for p in pairs]
        disease_gene_lists = [disease_to_genes.get(p[1], []) for p in pairs]

        self.drug_indices = self._pad_indices(drug_gene_lists, gene_to_idx)
        self.disease_indices = self._pad_indices(disease_gene_lists, gene_to_idx)

    def _pad_indices(self, all_gene_lists, gene_to_idx):
        # 找到最长的一个基因列表有多少个基因。
        max_len = max([len(l) for l in all_gene_lists]) if all_gene_lists else 1

        # 创建一个矩阵，默认填充 -1（作为空位的占位符）。
        matrix = np.full((len(all_gene_lists), max_len), -1, dtype=np.int32)

        for i, gene_list in enumerate(all_gene_lists):
            # 将基因名转为索引，只保留存在于字典中的基因。
            idxs = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
            if idxs:
                matrix[i, :len(idxs)] = idxs
        return matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 初始化一个全零的特征向量
        vec = torch.zeros(self.gene_dim * 2, dtype=torch.float32)

        # 获取当前样本对应的基因索引数组
        d_idx_row = self.drug_indices[idx]
        dis_idx_row = self.disease_indices[idx]

        # 快速填充：只给不是 -1 的位置填 1
        # 药物部分（前段）
        valid_d = d_idx_row[d_idx_row != -1]
        vec[valid_d] = 1.0

        # 疾病部分（后段）
        valid_dis = dis_idx_row[dis_idx_row != -1]
        vec[self.gene_dim + valid_dis] = 1.0

        return vec, torch.tensor(self.labels[idx])


# ==========================================
# 1. 定义 G2D2-Net 神经网络模型
# ==========================================
class G2D2Net(nn.Module):
    def __init__(self, input_dim):
        super(G2D2Net, self).__init__()  # 调用 nn.Module 的初始化函数。
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 降维
            nn.BatchNorm1d(1024),  # 批归一化层
            nn.ReLU(),  # ReLU 映射输出
            nn.Dropout(0.3),  # 防止过拟合

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. 训练和画图逻辑
# ==========================================
def plot_metrics(y_true, y_probs, y_preds):
    plt.figure(figsize=(18, 5))

    # 1. ROC Curve
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # 2. PR Curve
    plt.subplot(1, 3, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # 3. Confusion Matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('output/evaluation_plots.png', dpi=1200)
    print("\n[INFO] 评估图表已保存至 output/evaluation_plots.png")
    plt.close()


def main():
    print("正在加载数据...")
    with open('output/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    pairs = data['pairs']
    labels = data['labels']
    drug_to_genes = data['drug_to_genes']
    disease_to_genes = data['disease_to_genes']
    gene_to_idx = data['gene_to_idx']
    gene_dim = data['gene_dim']
    input_dim = gene_dim * 2

    # 划分训练集和测试集的索引，train : val : test = 70% : 15% : 15%。
    from sklearn.model_selection import train_test_split
    pairs_temp, pairs_test, labels_temp, labels_test = train_test_split(pairs, labels, test_size=0.15, random_state=42)
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(pairs_temp, labels_temp, test_size=0.1765,
                                                                        random_state=42)

    # 初始化模型、损失函数和优化器，这里可以根据显卡类型自动调整加速手段，本人代码在 CUDA 和 MPS 上跑，故加。
    def get_device():
        print(f"PyTorch 版本: {torch.__version__}")
        # 1. 检查 NVIDIA GPU (Windows/Linux/AMD-ROCm)
        if torch.cuda.is_available():
            # 甚至可以打印出显卡型号，方便调试。
            print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
            print(f"使用 Windows/Linux NVIDIA CUDA 进行运算。")
            return torch.device("cuda")

        # 2. 检查 Apple Silicon GPU (macOS)
        elif torch.backends.mps.is_available():
            print("检测到 Apple Silicon GPU (MPS)")
            print(f"使用 Apple MPS 进行加速运算。")
            return torch.device("mps")

        # 3. 可选：检查 Windows 上的 AMD/Intel 显卡 (需安装 torch-directml)
        # try:
        #     import torch_directml
        #     if torch_directml.is_available():
        #         return torch_directml.device()
        # except ImportError:
        #     pass

        # 4. 最终回退到 CPU
        print("只使用 CPU 进行运算。")
        return torch.device("cpu")

    device = get_device()

    is_cuda = device.type == "cuda"

    use_amp = is_cuda

    print(f"使用硬件: {device} | 是否使用 AMP 混合精度: {use_amp}")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 实例化动态数据集
    print("正在创建训练集，时间较长，请耐心等待。")
    train_dataset = G2D2Dataset(pairs_train, labels_train, drug_to_genes, disease_to_genes, gene_to_idx, gene_dim)

    print("正在创建验证集，时间较长，请耐心等待。")
    val_dataset = G2D2Dataset(pairs_val, labels_val, drug_to_genes, disease_to_genes, gene_to_idx, gene_dim)

    print("正在创建测试集，时间较长，请耐心等待。")
    test_dataset = G2D2Dataset(pairs_test, labels_test, drug_to_genes, disease_to_genes, gene_to_idx, gene_dim)

    # 试出来的血泪史！！！鬼知道我们被 Windows 环境下的 DataLoader 多进程机制的坑蹂躏了多久！！！最后索性用了 NumPy，只在最后转化成 tensor。
    # 没事别开多线程！！！会内存爆炸！！！我虚拟内存都开到 256GB 了都容易崩 MemoryError！！！
    run_num_workers = 0
    run_pin_memory = False

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=run_num_workers,
                              pin_memory=run_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=run_num_workers,
                            pin_memory=run_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=run_num_workers,
                             pin_memory=run_pin_memory)

    print("正在创建模型。")
    model = G2D2Net(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 优化器

    # ================= 训练阶段 =================
    epochs = 20
    best_val_acc = 0.0
    # 动态确定 AMP 设备：如果是 CUDA 就用 CUDA，否则用 CPU(针对 Mac/A 卡适配)。
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    print("\n================ 开始训练 G2D2-Net ================")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()  # 清空梯度

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(batch_X).squeeze()  # 得到预测后去掉多余维度
                loss = criterion(outputs, batch_y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)

        # ================= 验证阶段 =================
        model.eval()  # 关闭随机性
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"--> Epoch {epoch + 1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        hw_stats = get_hardware_stats(device)
        print(
            f"    [硬件状态] CPU: {hw_stats['cpu_percent']}% | RAM: {hw_stats['ram_used_gb']:.1f}/{hw_stats['ram_total_gb']:.1f} GB",
            end="")
        if 'gpu_allocated_mb' in hw_stats:
            print(
                f" | MPS 已分配: {hw_stats['gpu_allocated_mb']:.0f} MB (驱动占用: {hw_stats['gpu_driver_mb']:.0f} MB)")
        elif 'gpu_util_percent' in hw_stats:
            print(
                f" | GPU 利用率: {hw_stats['gpu_util_percent']}% | 显存: {hw_stats['gpu_mem_used_mb']:.0f}/{hw_stats['gpu_mem_total_mb']:.0f} MB")
        else:
            print()  # 换行

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'output/g2d2_best_model.pth')

    # ================= 测试阶段 =================
    print("\n================ 训练完成，开始在独立测试集上评估 ================")
    model.load_state_dict(torch.load('output/g2d2_best_model.pth'))
    model.eval()

    test_probs, test_preds, test_targets = [], [], []  # 概率、预测类别、真实标签
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="测试中"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()

            probs = torch.sigmoid(outputs)
            test_probs.extend(probs.cpu().numpy())
            preds = (probs >= 0.5).float()

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())

    # 计算各项指标
    acc = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds)
    roc_auc = roc_auc_score(test_targets, test_probs)
    pr_auc = average_precision_score(test_targets, test_probs)

    print("\n---------------- 测试集最终成绩 ----------------")
    print(f"Accuracy (准确率)    : {acc * 100:.2f}%")
    print(f"F1 Score (F1值)      : {f1:.4f}")
    print(f"ROC AUC (受试者工作曲线): {roc_auc:.4f}")
    print(f"PR AUC (均值精度)     : {pr_auc:.4f}")
    print("--------------------------------------------------")

    # 画图
    plot_metrics(test_targets, test_probs, test_preds)


if __name__ == '__main__':
    main()
