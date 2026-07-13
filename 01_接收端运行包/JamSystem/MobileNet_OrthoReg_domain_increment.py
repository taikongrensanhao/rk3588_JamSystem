import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from model_v2 import MobileNetV2
import itertools
from IQ_STFT_python import IQSTFTAnalyzer

analyzer = IQSTFTAnalyzer(fs=6.4e3, nperseg=1024, noverlap=512)

# ======================= 1. 正交正则化损失 (OrthoReg) =======================

def ortho_reg_loss(model, init_state_dict, lambda_ortho=0.001):
    """
    改进版：对每个层，计算相对误差（除以列数），然后取平均。
    """
    reg_loss = 0.0
    num_layers = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name and param.dim() >= 2:
            init_param = init_state_dict[name].to(param.device)
            delta = param - init_param
            # 展平为 2D
            if delta.dim() == 4:
                delta_2d = delta.view(delta.size(0), -1)   # [out, in*kh*kw]
            else:
                delta_2d = delta
            # 计算 Gram: (ΔW)^T ΔW
            gram = delta_2d.T @ delta_2d   # [d, d]
            identity = torch.eye(gram.size(0), device=gram.device)
            # 归一化：除以列数 d，使不同尺寸的层贡献相当
            d = gram.size(0)
            loss_layer = torch.norm(gram - identity, p='fro') ** 2 / d
            reg_loss += loss_layer
            num_layers += 1
    if num_layers > 0:
        reg_loss = reg_loss / num_layers   # 平均每层的损失
    return lambda_ortho * reg_loss


# ======================= 2. 模型定义（MobileNetV2，输出7类）======================
def get_model(num_classes=7, pretrained=True):
    # create model
    model = MobileNetV2(num_classes=num_classes, alpha=1)
    model_weight_path = '/root/autodl-tmp/Jam_recognization/MobileNet_OrthoReg/final_incremental_model.pth'
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path)  # 删掉了如下参数：, map_location='cpu'

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
    print(f"你的模型有，但预训练权重没有的层: {missing_keys}")
    print(f"预训练权重有，但你的模型不需要的层: {unexpected_keys}")
    return model


# ======================= 3. 数据加载（示例：模拟合成数据集）======================
# 实际使用时请根据您的数据格式替换以下 Dataset 类
def get_dataloader(domain_id, batch_size=128, batch_train=None, batch_test=None):
    """
    根据域 id 返回对应训练集和测试集的 DataLoader
    你需要实现自己的数据加载逻辑
    """
    batch_iq = []
    batch_labels = []
    metadata_df_qpsk = pd.read_csv(r'/root/autodl-tmp/Jam_recognization/Signal_Generate/rf_interference_dataset_qpsk/metadata.csv')
    data_dir_qpsk = r'/root/autodl-tmp/Jam_recognization/Signal_Generate/rf_interference_dataset_qpsk/'
    metadata_df_fm = pd.read_csv(r'/root/autodl-tmp/Jam_recognization/Signal_Generate/rf_interference_dataset_fm/metadata.csv')
    data_dir_fm = r'/root/autodl-tmp/Jam_recognization/Signal_Generate/rf_interference_dataset_fm'
    if batch_test == None:
        if domain_id=='QPSK':
            for i in range(7):
                for j in range(4):
                    for idx in range(batch_train):
                        row = metadata_df_qpsk.iloc[idx + j * 1000 + i * 4000]
                        print(row['iq_file'])
                        iq_data = np.load(os.path.join(data_dir_qpsk, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0,:], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)
        elif domain_id=='FM':
            for i in range(7):
                for j in range(4):
                    for idx in range(batch_train):
                        row = metadata_df_fm.iloc[idx + j * 1000 + i * 4000]
                        print(row['iq_file'])
                        iq_data = np.load(os.path.join(data_dir_fm, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0, :], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)

    else:
        if domain_id=='QPSK':
            for i in range(7):
                for j in range(4):
                    for idx in range(1000 - batch_test, 1000):
                        row = metadata_df_qpsk.iloc[idx + j * 1000 + i * 4000]
                        iq_data = np.load(os.path.join(data_dir_qpsk, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0,:], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)
        elif domain_id=='FM':
            for i in range(7):
                for j in range(4):
                    for idx in range(1000 - batch_test, 1000):
                        row = metadata_df_fm.iloc[idx + j * 1000 + i * 4000]
                        iq_data = np.load(os.path.join(data_dir_fm, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0, :], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)

    data_x = np.array(batch_iq)
    data_y = np.array(batch_labels)
    # 数据转化为(n_samples, channels, 1024, 81)格式
    data_x = np.reshape(data_x, (-1, 1, 1024, 81))
    # 1. 数据预处理：将数据转换为PyTorch需要的格式
    X_tensor = torch.FloatTensor(data_x)
    # 将one-hot标签转换为类别索引（PyTorch的CrossEntropyLoss需要类别索引）
    y_labels = torch.LongTensor(data_y)
    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# ======================= 4. 单域微调（带正交正则化） =======================
def train_task_vector(model, init_state_dict, train_loader, test_loader,
                      epochs=10, lr=1e-4, lambda_ortho=0.1, device='cuda'):
    """
    微调模型得到任务向量 τ = θ_t - θ_0
    返回: 微调后的模型, 任务向量字典（可包含状态字典）
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            task_loss = criterion(outputs, labels)
            # 正交正则化
            ortho_loss = ortho_reg_loss(model, init_state_dict, lambda_ortho)
            loss = task_loss + ortho_loss
            # loss = task_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    # 计算任务向量 (τ = θ_t - θ_0) 存储为状态字典的差
    task_vector = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name and param.dim() >= 2:
            task_vector[name] = param.data - init_state_dict[name].to(device)

    return model, task_vector, best_acc


# ======================= 5. 模型合并 =======================
def merge_models(init_state_dict, task_vectors, alpha):
    """
    合并多个任务向量: θ_merged = θ_0 + α * Σ τ_t
    task_vectors: list of dict, 每个dict存储一个任务的部分参数τ
    alpha: 缩放系数
    """
    merged_state = copy.deepcopy(init_state_dict)
    for name in merged_state.keys():
        # 只处理那些所有任务向量中都存在的参数（即我们存储过的权重）
        if name in task_vectors[0]:
            sum_tau = sum(tv[name] for tv in task_vectors)
            merged_state[name] = init_state_dict[name].to(sum_tau.device) + alpha * sum_tau
        # 其他参数（如偏置、BN的weight/bias）保持不变，使用预训练初始值
    return merged_state


def evaluate_model(model, test_loaders, device='cuda'):
    """
    评估合并后模型在多个域上的表现。
    test_loaders: dict, {domain_name: test_loader}
    返回: 每个域的准确率字典，平均准确率
    """
    model.eval()
    accuracies = {}
    for domain, loader in test_loaders.items():
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        accuracies[domain] = acc
    avg_acc = np.mean(list(accuracies.values()))
    return accuracies, avg_acc


def find_best_alpha(init_state_dict, task_vectors, test_loaders, alphas, device='cuda'):
    """
    网格搜索最佳 alpha
    """
    best_alpha = 0.0
    best_avg_acc = 0.0
    for alpha in alphas:
        merged_state = merge_models(init_state_dict, task_vectors, alpha)
        # 临时创建一个模型来评估
        model = get_model(num_classes=7, pretrained=False)
        model.load_state_dict(merged_state, strict=False)
        model = model.to(device)
        _, avg_acc = evaluate_model(model, test_loaders, device)
        print(f"Alpha={alpha:.3f} -> Average Accuracy: {avg_acc:.4f}")
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_alpha = alpha
    return best_alpha, best_avg_acc


# ======================= 6. 主流程 =======================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 配置参数
    num_classes = 7
    batch_size = 128
    epochs = 50
    lr = 1e-4
    lambda_ortho = 10   # 正则化强度，可根据实验调整
    alphas = np.arange(0.0, 1.05, 0.05)  # [0.0, 0.05, ..., 1.0]

    # 定义域列表（调制方式）
    domains = ['FM']  # 示例['QPSK', 'FM', '16QAM']，根据实际数据修改
    # 为每个域准备数据加载器（实际应用中需提供对应路径）
    # 这里模拟为每个域创建独立的 DataLoader（实际需替换为真实数据目录）
    train_loaders = {}
    test_loaders = {}
    for domain in domains:
        train_loader = get_dataloader(domain, batch_size=batch_size, batch_train=800)
        test_loader = get_dataloader(domain, batch_size=batch_size, batch_test=200)
        train_loaders[domain] = train_loader
        test_loaders[domain] = test_loader

    # 初始化预训练模型 (θ_0)
    init_model = get_model(num_classes=num_classes, pretrained=True)
    init_state_dict = copy.deepcopy(init_model.state_dict())

    # 存储每个域微调后的任务向量 τ_t
    task_vectors = []
    domain_accs = {}

    # 阶段1: 对每个域分别微调，获取任务向量
    for domain in domains:
        print(f"\n===== Training on domain: {domain} =====")
        model = get_model(num_classes=num_classes, pretrained=True)  # 从预训练开始
        model.load_state_dict(init_state_dict)  # 确保起点一致
        model, task_vector, final_acc = train_task_vector(
            model, init_state_dict,
            train_loaders[domain], test_loaders[domain],
            epochs=epochs, lr=lr, lambda_ortho=lambda_ortho, device=device
        )
        task_vectors.append(task_vector)
        domain_accs[domain] = final_acc
        print(f"Domain {domain} single-task accuracy: {final_acc:.4f}")

    # 阶段2: 合并任务向量，搜索最佳 α
    print("\n===== Searching best alpha for merging =====")
    best_alpha, best_avg_acc = find_best_alpha(
        init_state_dict, task_vectors, test_loaders, alphas, device
    )
    print(f"Best alpha: {best_alpha:.3f}, Best average accuracy: {best_avg_acc:.4f}")

    # 使用最佳 alpha 构建最终合并模型
    final_merged_state = merge_models(init_state_dict, task_vectors, best_alpha)
    final_model = get_model(num_classes=num_classes, pretrained=False)
    final_model.load_state_dict(final_merged_state, strict=False)
    final_model = final_model.to(device)

    # 最终评估
    print("\n===== Final merged model evaluation =====")
    final_accs, final_avg = evaluate_model(final_model, test_loaders, device)
    for domain, acc in final_accs.items():
        print(f"Domain {domain}: {acc:.4f}")
    print(f"Average accuracy: {final_avg:.4f}")

    # 可选：保存合并后的模型
    torch.save(final_merged_state, "final_merged_model.pth")
    print("Merged model saved to merged_model.pth")


if __name__ == "__main__":
    main()