import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import SST2
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torchtext.transforms import VocabTransform, ToTensor
from torch.nn.utils.rnn import pad_sequence

# 使用 GloVe 词向量
glove = GloVe(name='6B', dim=100)

# 文本和标签处理器
class TextProcessor:
    def __init__(self, vocab, max_length=200):
        self.vocab_transform = VocabTransform(vocab)
        self.max_length = max_length
        self.to_tensor = ToTensor()

    def __call__(self, text):
        tokens = text.lower().split()  # 分词并转换为小写
        tokens = self.vocab_transform(tokens)  # 将单词映射为词汇索引
        return self.to_tensor(tokens[:self.max_length])  # 转为张量并截断到 max_length

def label_processor(label):
    return torch.tensor(int(label))  # 将标签转换为张量

# 自定义批处理函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [text_processor(text) for text in texts]
    texts = pad_sequence(texts, batch_first=True)  # 填充到相同长度
    labels = torch.stack([label_processor(label) for label in labels])
    return texts, labels

# 加载 SST2 数据集
train_data = list(SST2(split='train'))
valid_data = list(SST2(split='validation'))
test_data = list(SST2(split='test'))

# 创建 DataLoader
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 模型类
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(glove.vectors)
        self.embedding.weight.requires_grad = False  # GloVe 词向量不训练

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead=2, dim_feedforward=hidden_dim), num_layers
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)  # 使用最大池化
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        pooled = self.pooling(encoded.permute(0, 2, 1)).squeeze(-1)
        return self.classifier(pooled)

# 创建模型、优化器、损失函数
vocab_size = len(glove.stoi)
embedding_dim = 100
hidden_dim = 128
num_layers = 2
num_classes = 2  # SST2 是二分类任务

model = TextClassificationModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for texts, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

# 验证函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            predictions = model(texts)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            predicted_labels = predictions.argmax(1)
            correct += (predicted_labels == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# 训练模型
N_EPOCHS = 16
for epoch in range(N_EPOCHS):
    train_model(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion)
    print(f'Epoch: {epoch+1:02}, Validation Loss: {valid_loss:.3f}, Validation Acc: {valid_acc*100:.2f}%')

# 测试模型
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')
