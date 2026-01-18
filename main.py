import torch
import torch.nn as nn
from transformers import GraphCodeBERTModel, DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 加载预训练模型和分词器
graphcodebert_model = GraphCodeBERTModel.from_pretrained("microsoft/graphcodebert-base")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # 计算每个脚本的权重
        weighted_sum = torch.sum(weights * x, dim=1)  # 使用加权和来获得最终的嵌入
        return weighted_sum


# 获取代码内容的嵌入（使用DistilBERT）
def get_code_embeddings(code):
    inputs = distilbert_tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    embeddings = distilbert_model(**inputs).last_hidden_state
    return embeddings


# 获取依赖图的嵌入（使用PyCG或类似方法获取实际的图嵌入）
def get_dependency_embeddings(code):
    # 占位符：实际实现应使用PyCG来提取依赖图
    return torch.rand(1, 768)  # 目前返回随机嵌入


# 定义代码库嵌入生成器与注意力机制
class RepositoryEmbedder(nn.Module):
    def __init__(self, input_size):
        super(RepositoryEmbedder, self).__init__()
        self.attention_layer = Attention(input_size)

    def forward(self, code_embeddings, dependency_embeddings):
        # 合并代码嵌入和依赖图嵌入
        combined_embeddings = torch.cat((code_embeddings, dependency_embeddings), dim=1)
        # 使用注意力机制处理嵌入
        final_embedding = self.attention_layer(combined_embeddings)
        return final_embedding


# 定义多标签分类网络
class RepositoryClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(RepositoryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # 全连接层用于分类
        self.sigmoid = nn.Sigmoid()  # 使用sigmoid函数进行多标签分类

    def forward(self, x):
        return self.sigmoid(self.fc(x))  # 输出标签概率


# 定义数据集类（实际使用时需要替换为真实的代码库数据）
class RepositoryDataset(Dataset):
    def __init__(self, code_samples, labels):
        self.code_samples = code_samples  # 代码样本
        self.labels = labels  # 标签

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code_sample = self.code_samples[idx]
        label = self.labels[idx]
        code_embeddings = get_code_embeddings(code_sample)  # 获取代码嵌入
        dependency_embeddings = get_dependency_embeddings(code_sample)  # 获取依赖图嵌入
        return code_embeddings, dependency_embeddings, label


# 示例数据（替换为实际的GitHub代码库数据和多标签话题）
code_samples = ["def add(a, b): return a + b", "class MyClass: pass"]  # 示例代码
labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]  # 多标签话题分配

# 创建数据集和数据加载器
dataset = RepositoryDataset(code_samples, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化模型
repository_embedder = RepositoryEmbedder(input_size=1536)  # 代码+依赖嵌入（768+768）
repository_classifier = RepositoryClassifier(input_size=768, output_size=5)  # 示例：5个话题

# 优化器和损失函数
optimizer = optim.Adam(list(repository_embedder.parameters()) + list(repository_classifier.parameters()), lr=0.001)
criterion = nn.BCELoss()  # 多标签分类的二元交叉熵损失

# 训练循环
for epoch in range(10):  # 示例：10轮训练
    repository_embedder.train()  # 设置为训练模式
    repository_classifier.train()
    running_loss = 0.0

    for code_embeddings, dependency_embeddings, label in dataloader:
        # 前向传播
        code_embeddings = code_embeddings.squeeze(1)  # 去除batch维度
        dependency_embeddings = dependency_embeddings.squeeze(1)
        label = label.float()  # 将标签转换为浮动类型

        # 生成代码库级嵌入
        repository_embedding = repository_embedder(code_embeddings, dependency_embeddings)

        # 分类代码库话题
        outputs = repository_classifier(repository_embedding)

        # 计算损失
        loss = criterion(outputs, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"第{epoch + 1}轮训练，损失值：{running_loss / len(dataloader)}")

# 示例推理
repository_classifier.eval()  # 设置为评估模式
code_sample = "def add(a, b): return a + b"  # 示例代码
code_embeddings = get_code_embeddings(code_sample)
dependency_embeddings = get_dependency_embeddings(code_sample)
repository_embedding = repository_embedder(code_embeddings, dependency_embeddings)
topic_scores = repository_classifier(repository_embedding)

print("话题分类分数:", topic_scores)
