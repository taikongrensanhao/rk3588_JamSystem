import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from model_v2 import MobileNetV2
from IQ_STFT_python import IQSTFTAnalyzer

analyzer = IQSTFTAnalyzer(fs=6.4e3, nperseg=1024, noverlap=512)


# ======================= 1. 正交正则化损失 (OrthoReg) =======================
def ortho_reg_loss(model, init_state_dict, lambda_ortho=0.1):
    """
    计算正交正则化损失，自动处理形状不匹配（如分类器扩展后的新行）。
    """
    reg_loss = 0.0
    num_layers = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name and param.dim() >= 2:
            # 获取初始参数（可能形状不同）
            if name in init_state_dict:
                init_param = init_state_dict[name].to(param.device)
                # 若形状不同，尝试对齐（例如分类器扩展时 init 少了几行，补零）
                if init_param.shape != param.shape:
                    # 假设 param 比 init_param 多出若干行（新类别）
                    if param.dim() == 2 and param.size(0) > init_param.size(0):
                        # 分类器权重: [new_out, in_features] vs [old_out, in_features]
                        padded = torch.zeros_like(param)
                        padded[:init_param.size(0)] = init_param
                        init_param = padded
                    elif param.dim() == 4 and param.size(0) > init_param.size(0):
                        # 卷积层输出通道增加（较少见，但处理一下）
                        padded = torch.zeros_like(param)
                        padded[:init_param.size(0)] = init_param
                        init_param = padded
                    else:
                        # 其他情况，跳过该层（或直接使用零增量）
                        continue
            else:
                # 初始状态字典中没有该层（全新增参数），则 ΔW = param - 0
                init_param = torch.zeros_like(param)

            delta = param - init_param
            # 展平为 2D
            if delta.dim() == 4:  # Conv2d
                delta_2d = delta.view(delta.size(0), -1)
            else:
                delta_2d = delta
            # 计算 Gram 矩阵
            gram = delta_2d.T @ delta_2d
            d = gram.size(0)
            identity = torch.eye(d, device=gram.device, dtype=gram.dtype)
            loss_layer = torch.norm(gram - identity, p='fro') ** 2 / d
            reg_loss += loss_layer
            num_layers += 1
    if num_layers > 0:
        reg_loss /= num_layers
    return lambda_ortho * reg_loss


# ======================= 2. 模型定义与分类器扩展 =======================
def get_base_model(num_classes, weights_path):
    """
    创建 MobileNetV2 并加载在初始类别上训练好的权重。
    """
    model = MobileNetV2(num_classes=num_classes, alpha=1)
    assert os.path.exists(weights_path), f"Weight file {weights_path} not found."
    state = torch.load(weights_path, map_location='cpu')
    # 只加载匹配的层（忽略分类器维数不匹配的警告）
    model.load_state_dict(state, strict=False)
    return model


def expand_classifier(model, num_new_classes):
    """
    扩展 MobileNetV2 最后一层分类器的输出维度。
    新增的权重行初始化为 0，偏置初始化为 0。
    """
    old_linear = model.classifier[1]
    old_out = old_linear.out_features
    new_out = old_out + num_new_classes
    new_weight = torch.zeros((new_out, old_linear.in_features), device=old_linear.weight.device)
    new_weight[:old_out] = old_linear.weight.data
    new_bias = torch.zeros(new_out, device=old_linear.bias.device)
    new_bias[:old_out] = old_linear.bias.data
    new_linear = nn.Linear(old_linear.in_features, new_out)
    new_linear.weight.data = new_weight
    new_linear.bias.data = new_bias
    model.classifier[1] = new_linear
    return model


# ======================= 3. 数据加载（类增量版本） =======================
def get_new_class_dataloader(domain_id, batch_size=128, batch_train=None, batch_test=None):
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
            for i in range(1):
                for j in range(4):
                    for idx in range(batch_train):
                        row = metadata_df_qpsk.iloc[idx + j * 1000 + (i+6) * 4000]
                        print(row['iq_file'])
                        iq_data = np.load(os.path.join(data_dir_qpsk, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0,:], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)
        elif domain_id=='FM':
            for i in range(1):
                for j in range(4):
                    for idx in range(batch_train):
                        row = metadata_df_fm.iloc[idx + j * 1000 + (i+6) * 4000]
                        print(row['iq_file'])
                        iq_data = np.load(os.path.join(data_dir_fm, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0, :], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)

    else:
        if domain_id=='QPSK':
            for i in range(1):
                for j in range(4):
                    for idx in range(1000 - batch_test, 1000):
                        row = metadata_df_qpsk.iloc[idx + j * 1000 + (i+6) * 4000]
                        iq_data = np.load(os.path.join(data_dir_qpsk, 'iq_data', row['iq_file']))
                        # 进行STFT分析
                        results = analyzer.stft_analysis(iq_data[0,:], mode='all')
                        batch_iq.append(results['magnitude_db'])
                        batch_labels.append(i)
        elif domain_id=='FM':
            for i in range(1):
                for j in range(4):
                    for idx in range(1000 - batch_test, 1000):
                        row = metadata_df_fm.iloc[idx + j * 1000 + (i+6) * 4000]
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


def get_all_class_dataloader(domain_id, batch_size=128, batch_train=None, batch_test=None):
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


# ======================= 4. 训练单个任务向量（新类） =======================
def train_task_vector(base_model, init_state_dict, new_class_train_loader, new_class_test_loader,
                      new_class_idx, epochs=20, lr=1e-4, lambda_ortho=1.0, device='cuda'):
    """
    在基础模型上增加一个新类，训练任务向量。

    Args:
        base_model: 基础模型（未扩展分类器，输出维度 = 当前已知类别数）
        init_state_dict: 基础模型的初始状态字典（θ0）
        new_class_loader: 仅包含新类样本的 DataLoader（标签为原始类别索引，如 3）
        test_loaders_dict: 字典 {domain: test_loader}，用于验证（包含所有已知类）
        new_class_idx: 新类的全局索引（例如当前已知类别数为3，则新类索引为3）
        epochs, lr, lambda_ortho: 训练超参数
    Returns:
        task_vector: 字典，包含所有权重的更新量（包括扩展的分类器）
        tuned_model: 微调后的模型（已扩展分类器）
        best_acc: 在验证集上的最佳准确率（所有已知类）
    """
    # 1. 复制基础模型并扩展分类器
    model = copy.deepcopy(base_model).to(device)
    # 扩展输出维度：新增加 1 个类
    model = expand_classifier(model, num_new_classes=1)

    # 注意：扩展后模型输出维度 = old_out + 1
    # 构建扩展后的初始状态字典（用于正交正则化和任务向量计算）
    expanded_init = copy.deepcopy(init_state_dict)
    old_out = base_model.classifier[1].out_features
    new_out = model.classifier[1].out_features
    # 扩展权重
    old_w = expanded_init['classifier.1.weight'].to(device)
    new_w = torch.zeros((new_out, old_w.size(1)), device=device, dtype=old_w.dtype)
    new_w[:old_out] = old_w
    expanded_init['classifier.1.weight'] = new_w
    # 扩展偏置
    old_b = expanded_init['classifier.1.bias'].to(device)
    new_b = torch.zeros(new_out, device=device, dtype=old_b.dtype)
    new_b[:old_out] = old_b
    expanded_init['classifier.1.bias'] = new_b

    # 2. 重新映射标签：新类样本的标签由原来的 `new_class_idx` 变为扩展后的最大索引
    #    例如原输出为 [0,1,2]，扩展后输出为 [0,1,2,3]，新类样本应标为 3
    #    我们创建一个新的 DataLoader，动态修改标签
    def remap_labels(loader, new_label):
        for x, y in loader:
            # 将所有样本的标签替换为 new_label（因为该 loader 只包含新类）
            y_new = torch.full_like(y, new_label)
            yield x, y_new

    # new_train_loader = remap_labels(new_class_train_loader, model.classifier[1].out_features - 1)
    # new_test_loader = remap_labels(new_class_test_loader, model.classifier[1].out_features - 1)

    # 3. 优化器与损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        new_train_loader = remap_labels(new_class_train_loader, model.classifier[1].out_features - 1)
        new_test_loader = remap_labels(new_class_test_loader, model.classifier[1].out_features - 1)
        for images, labels in tqdm(new_train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            # 现在计算正交损失
            ortho_loss = ortho_reg_loss(model, expanded_init, lambda_ortho)
            loss = ce_loss + ortho_loss
            # loss = ce_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in new_test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 计算任务向量 τ = θ_tuned - θ0（θ0 需扩展至相同结构）
    task_vector = {}
    for name, param in model.state_dict().items():
        if 'weight' in name or 'bias' in name:
            if name in expanded_init:
                task_vector[name] = param - expanded_init[name].to(device)
            else:
                task_vector[name] = param.clone()
    return task_vector, model, best_acc


# ======================= 5. 合并任务向量 =======================
def merge_models(init_state_dict, task_vectors, alpha):
    """
    合并多个任务向量，自动扩展 init_state_dict 以匹配任务向量形状。
    """
    # 首先，确定目标设备
    target_device = torch.device('cpu')

    # 1. 扩展 init_state_dict，使其形状与 task_vectors 中的参数对齐
    expanded_init = copy.deepcopy(init_state_dict)
    for tv in task_vectors:
        for name, delta in tv.items():
            # 只处理权重和偏置
            if 'weight' not in name and 'bias' not in name:
                continue
            # 如果 init 中没有这个参数，说明是新增的（如扩展的分类器行），初始化为零
            if name not in expanded_init:
                expanded_init[name] = torch.zeros_like(delta)
            else:
                init_param = expanded_init[name]
                # 如果形状不同，进行填充
                if init_param.shape != delta.shape:
                    # 处理权重矩阵（2D）或卷积核（4D）的输出维度扩展
                    if delta.dim() == 2 and delta.size(0) > init_param.size(0):
                        # 分类器权重: [new_out, in_features] vs [old_out, in_features]
                        padded = torch.zeros_like(delta)
                        padded[:init_param.size(0)] = init_param
                        expanded_init[name] = padded
                    elif delta.dim() == 1 and delta.size(0) > init_param.size(0):
                        # 偏置
                        padded = torch.zeros_like(delta)
                        padded[:init_param.size(0)] = init_param
                        expanded_init[name] = padded
                    else:
                        # 其他情况（理论上不应发生），跳过
                        continue
    # 将 init_state_dict 中的所有参数移到 CPU
    init_cpu = {k: v.to(target_device) for k, v in expanded_init.items()}
    # 2. 执行合并：θ_merged = expanded_init + α * Σ τ_t
    merged_state = copy.deepcopy(init_cpu)
    for tv in task_vectors:
        for name, delta in tv.items():
            if 'weight' not in name and 'bias' not in name:
                continue
            delta = delta.to(target_device)
            # 类型统一为 float
            if merged_state[name].dtype != delta.dtype:
                merged_state[name] = merged_state[name].to(delta.dtype)
            merged_state[name] += alpha * delta

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

def find_best_alpha(init_state_dict, base_model_path, task_vectors, current_total_classes, test_loaders, alphas, device='cuda'):
    """
    网格搜索最佳 alpha
    """
    best_alpha = 0.0
    best_avg_acc = 0.0
    for alpha in alphas:
        merged_state = merge_models(init_state_dict, task_vectors, alpha)
        merged_model = get_base_model(6, base_model_path)  # 先创建基础结构
        # 扩展分类器至当前总数
        for _ in range(len(task_vectors)):
            merged_model = expand_classifier(merged_model, 1)
        merged_model.load_state_dict(merged_state, strict=False)
        merged_model = merged_model.to(device)

        # 评估当前合并模型在所有已知类上的性能
        print("\nEvaluating merged model after adding class")
        domain_accs, avg_acc = evaluate_model(merged_model, test_loaders, device)
        for dom, acc in domain_accs.items():
            print(f"  {dom}: {acc:.4f}")
        print(f"  Average accuracy: {avg_acc:.4f}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_alpha = alpha
    return best_alpha, best_avg_acc


# ======================= 6. 主流程：类增量学习 =======================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ------------- 配置参数 -------------
    initial_classes = 6  # 基础模型已训练的类别数（例如0,1,2）
    new_classes_list = [6]  # 按顺序新增的类别索引（共3个新类）
    domains = ['QPSK']  # 使用的调制方式域

    batch_size = 128
    epochs_per_task = 50
    lr = 1e-4
    lambda_ortho = 10  # 正交正则化强度，可根据实验调整
    alphas = np.arange(0.0, 1.05, 0.05)  # 合并系数搜索范围
    # 基础模型路径（已训练好 initial_classes 类）
    base_model_path = '/root/autodl-tmp/Jam_recognization/mobilenet/MobileNetV2_QPSKDataTrain_6classes_IncludeNone.pth'  # 请替换为实际路径

    # ------------- 1. 加载基础模型 θ0 -------------
    base_model = get_base_model(num_classes=initial_classes, weights_path=base_model_path)
    init_state_dict = copy.deepcopy(base_model.state_dict())

    # ------------- 2. 准备测试数据（所有类） -------------
    # 构建包含所有类的测试集（用于合并模型性能）
    test_loaders = {}
    for domain in domains:
        # train_loader = get_all_class_dataloader(domain, batch_size=batch_size, batch_train=800)
        test_loader = get_all_class_dataloader(domain, batch_size=batch_size, batch_test=200)
        # train_loaders[domain] = train_loader
        test_loaders[domain] = test_loader

    # 存储所有任务向量
    task_vectors = []
    current_total_classes = initial_classes

    # 增量学习循环
    for new_class_idx in new_classes_list:
        print(f"\n===== Adding new class: {new_class_idx} =====")

        # 准备该新类的训练数据（使用某个域，这里以 QPSK 为例，可扩展）
        # 实际应用中可能多个域的数据混合，这里简化
        train_loader_new = get_new_class_dataloader(domains[0], batch_size=128, batch_train=800)
        test_loader_new = get_new_class_dataloader(domains[0], batch_size=128, batch_test=200)

        # 训练任务向量
        task_vector, tuned_model, best_new_acc = train_task_vector(
            base_model, init_state_dict, train_loader_new, test_loader_new,
            new_class_idx, epochs=epochs_per_task, lr=lr,
            lambda_ortho=lambda_ortho, device=device
        )
        print(f"Task vector learned. New classes avg acc after training: {best_new_acc:.4f}")

        # 存储任务向量
        task_vectors.append(task_vector)
        # 更新当前总数
        current_total_classes += 1

    # 合并所有任务向量（搜索最佳alpha）
    best_alpha, best_avg_acc = find_best_alpha(
        init_state_dict, base_model_path, task_vectors, current_total_classes, test_loaders, alphas, device
    )
    print(f"Best alpha: {best_alpha:.3f}, Best average accuracy: {best_avg_acc:.4f}")

    final_merged_state = merge_models(init_state_dict, task_vectors, best_alpha)
    final_model = get_base_model(initial_classes, base_model_path)
    for _ in range(len(new_classes_list)):
        final_model = expand_classifier(final_model, 1)
    final_model.load_state_dict(final_merged_state, strict=False)
    final_model = final_model.to(device)

    # 评估当前合并模型在所有已知类上的性能
    print("\nEvaluating merged model after adding class")
    domain_accs, avg_acc = evaluate_model(final_model, test_loaders, device)
    for dom, acc in domain_accs.items():
        print(f"  {dom}: {acc:.4f}")
    print(f"  Average accuracy: {avg_acc:.4f}")
    # 最终保存合并模型
    torch.save(final_model.state_dict(), "final_incremental_model.pth")
    print("\nFinal model saved to final_incremental_model.pth")


if __name__ == "__main__":
    main()