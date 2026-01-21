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
        weights = torch.softmax(self.attention(x), dim=1)
        weighted_sum = torch.sum(weights * x, dim=1)
        return weighted_sum


# 获取代码内容的嵌入
def get_code_embeddings(code):
    inputs = distilbert_tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    embeddings = distilbert_model(**inputs).last_hidden_state
    return embeddings


# 获取依赖图的嵌入
def get_dependency_embeddings(code):
    return torch.rand(1, 768)


# 定义代码库嵌入生成器与注意力机制
class RepositoryEmbedder(nn.Module):
    def __init__(self, input_size):
        super(RepositoryEmbedder, self).__init__()
        self.attention_layer = Attention(input_size)

    def forward(self, code_embeddings, dependency_embeddings):
        combined_embeddings = torch.cat((code_embeddings, dependency_embeddings), dim=1)
        final_embedding = self.attention_layer(combined_embeddings)
        return final_embedding


# 定义多标签分类网络
class RepositoryClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(RepositoryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


# 定义数据集类
class RepositoryDataset(Dataset):
    def __init__(self, code_samples, labels):
        self.code_samples = code_samples
        self.labels = labels

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code_sample = self.code_samples[idx]
        label = self.labels[idx]
        code_embeddings = get_code_embeddings(code_sample)
        dependency_embeddings = get_dependency_embeddings(code_sample)
        return code_embeddings, dependency_embeddings, label


code_samples = ["def add(a, b): return a + b", "class MyClass: pass"]
labels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]

dataset = RepositoryDataset(code_samples, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

repository_embedder = RepositoryEmbedder(input_size=1536)
repository_classifier = RepositoryClassifier(input_size=768, output_size=5)

optimizer = optim.Adam(list(repository_embedder.parameters()) + list(repository_classifier.parameters()), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    repository_embedder.train()
    repository_classifier.train()
    running_loss = 0.0

    for code_embeddings, dependency_embeddings, label in dataloader:
        code_embeddings = code_embeddings.squeeze(1)
        dependency_embeddings = dependency_embeddings.squeeze(1)
        label = label.float()

        repository_embedding = repository_embedder(code_embeddings, dependency_embeddings)

        outputs = repository_classifier(repository_embedding)

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"第{epoch + 1}轮训练，损失值：{running_loss / len(dataloader)}")

repository_classifier.eval()
code_sample = "def add(a, b): return a + b"
code_embeddings = get_code_embeddings(code_sample)
dependency_embeddings = get_dependency_embeddings(code_sample)
repository_embedding = repository_embedder(code_embeddings, dependency_embeddings)
topic_scores = repository_classifier(repository_embedding)

print("话题分类分数:", topic_scores)
