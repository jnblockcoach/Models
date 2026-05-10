import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleChatModel(nn.Module):
    """
    轻量级对话模型
    架构: Embedding + GRU + Linear
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1):
        super(SimpleChatModel, self).__init__()
        
        # 嵌入层: 将词索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU层: 处理序列信息
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层: 预测下一个词
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.3)
        
        # 初始化权重
        self._init_weights()
        
        # 打印参数数量
        self._print_params_count()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
        # 初始化GRU权重
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def _print_params_count(self):
        """打印模型参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"模型参数统计:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        
        # 各层参数详情
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        gru_params = sum(p.numel() for p in self.gru.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        print(f"  - Embedding层: {embedding_params:,}")
        print(f"  - GRU层: {gru_params:,}")
        print(f"  - FC层: {fc_params:,}")
    
    def forward(self, x, hidden=None):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len]
            hidden: 初始隐藏状态 [num_layers, batch_size, hidden_dim]
        Returns:
            output: 输出 [batch_size, seq_len, vocab_size]
            hidden: 最终隐藏状态
        """
        # 嵌入
        x = self.embedding(x)
        x = self.dropout(x)
        
        # GRU
        output, hidden = self.gru(x, hidden)
        
        # 输出层
        output = self.fc(output)
        
        return output, hidden
    
    def generate(self, input_ids, max_length=20, temperature=0.8, top_k=50):
        """
        生成回复
        Args:
            input_ids: 输入的问题token序列 [1, seq_len]
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: 只从概率最高的k个token中采样
        Returns:
            generated_ids: 生成的token序列
        """
        self.eval()
        
        with torch.no_grad():
            # 获取初始隐藏状态
            _, hidden = self.forward(input_ids)
            
            # 使用最后一个token作为起始
            current_token = input_ids[:, -1:]
            generated_ids = []
            
            # 获取词汇表大小
            vocab_size = self.fc.out_features
            
            for _ in range(max_length):
                # 前向传播
                output, hidden = self.forward(current_token, hidden)
                
                # 获取最后一个时间步的输出
                logits = output[:, -1, :] / temperature
                
                # 确保 top_k 不超过词汇表大小
                actual_top_k = min(top_k, vocab_size)
                
                # Top-k采样
                if actual_top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, actual_top_k)
                    # 创建掩码，只保留top-k的token
                    mask = torch.full_like(logits, float('-inf'))
                    mask.scatter_(1, top_k_indices, top_k_values)
                    logits = mask
                
                # 计算概率并采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 如果生成了结束符，停止 (EOS token id = 3)
                if next_token.item() == 3:
                    break
                
                # 避免生成PAD token (id=0)
                if next_token.item() == 0:
                    continue
                    
                generated_ids.append(next_token.item())
                current_token = next_token
            
        self.train()
        return generated_ids


class Config:
    """模型配置 - 约1M参数"""
    def __init__(self):
        self.embedding_dim = 128   # 词嵌入维度
        self.hidden_dim = 256      # 隐藏层维度
        self.num_layers = 2       # GRU层数
        self.dropout = 0.3
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.num_epochs = 100
        self.max_seq_len = 20
        self.train_ratio = 0.9