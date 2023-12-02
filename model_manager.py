import pandas as pd
import torch
from model import SingleCellClassifier
import requests
from io import StringIO


# 重新定义加载模型的函数
def load_pytorch_model(model_path, num_features, num_classes):
    # 创建模型实例
    print("Creating model instance...")
    model = SingleCellClassifier(num_features, num_classes)

    # 加载模型状态字典
    print(f"Loading model state dictionary from {model_path}...")
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    print("Model state dictionary loaded successfully.")

    model.eval()
    print("Model set to evaluation mode.")
    return model


class ModelManager:
    def __init__(self):
        self.models = {}
        self.genes_lists = {}
        self.model_urls = {
            'Kidney': 'https://example.com/path_to_kidney_model.pth',
            'Retina': 'https://example.com/path_to_retina_model.pt'
        }
        self.genes_list_urls = {
            'Kidney': 'https://example.com/path_to_kidney_genes_list.csv',
            'Retina': 'https://example.com/path_to_retina_genes_list.csv'
        }
        self.model_params = {
            'Kidney': {'num_features': 6635, 'num_classes': 10},
            'Retina': {'num_features': 6635, 'num_classes': 10}
        }

    def download_file(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def load_model_and_genes_list(self, organ_type):
        if organ_type not in self.models:
            model_url = self.model_urls.get(organ_type)
            genes_list_url = self.genes_list_urls.get(organ_type)
            model_params = self.model_params.get(organ_type)

            if model_url and genes_list_url and model_params:
                # 从网上下载模型
                print(f"Downloading model for {organ_type}...")
                model_data = self.download_file(model_url)

                # 从网上下载基因列表
                print(f"Downloading genes list for {organ_type}...")
                genes_list_data = self.download_file(genes_list_url)

                # 加载模型
                model_path = f"temporary_{organ_type}_model.pth"
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                self.models[organ_type] = load_pytorch_model(model_path, **model_params)

                # 加载基因列表
                genes_list_str = genes_list_data.decode('utf-8')
                genes_list_df = pd.read_csv(StringIO(genes_list_str), header=0, usecols=[1])
                self.genes_lists[organ_type] = genes_list_df.squeeze().tolist()
            else:
                raise ValueError(f"No model, genes list URL, or model parameters defined for organ type '{organ_type}'")

    def get_model(self, organ_type):
        # 获取模型，如果未加载则进行加载
        if organ_type not in self.models:
            self.load_model_and_genes_list(organ_type)
        return self.models[organ_type]

    def get_genes_list(self, organ_type):
        # 获取基因列表，如果未加载则进行加载
        if organ_type not in self.genes_lists:
            self.load_model_and_genes_list(organ_type)
        return self.genes_lists[organ_type]
