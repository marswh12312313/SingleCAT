import scanpy as sc
import numpy as np
import torch
from scipy import sparse
from model_manager import ModelManager
from torch.utils.data import TensorDataset, DataLoader


def predict_with_loader(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移到相应的设备
    model.eval()  # 确保模型在评估模式
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)  # 确保数据在相同的设备上

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())

    return predictions


def predict_cell_types(adata, organ_type):
    model_manager = ModelManager()
    model = model_manager.get_model(organ_type)
    genes_list = model_manager.get_genes_list(organ_type)

    # 动态计算模型需要的特征数量
    required_features = len(genes_list)

    # 从AnnData对象中提取和处理基因表达数据
    adata_filtered = adata[:, adata.var_names.isin(genes_list)]

    # 将稀疏矩阵转换为稠密矩阵
    X_dense = adata_filtered.X.toarray() if sparse.issparse(adata_filtered.X) else adata_filtered.X

    # 计算并添加零填充列（如果需要）
    current_features = X_dense.shape[1]
    missing_features = required_features - current_features
    if missing_features > 0:
        padding = np.zeros((adata_filtered.shape[0], missing_features))
        X_dense = np.hstack((X_dense, padding))

    # 转换为PyTorch Tensor，确保形状正确
    X_tensor = torch.from_numpy(X_dense).float()

    # 检查形状并添加通道维度（如果需要）
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)  # 添加通道维度，形状变为 [batch_size, 1, length]

    # 封装为 TensorDataset
    test_dataset = TensorDataset(X_tensor)

    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 使用 DataLoader 进行预测
    prediction_results = predict_with_loader(model, test_loader)

    # 将预测结果添加到AnnData对象的.obs属性中
    adata.obs['predicted_cell_type'] = prediction_results

    return adata
