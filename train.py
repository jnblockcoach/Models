import os
# 创建必要文件夹 - 必须在logging之前！
os.makedirs('data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
from model import SimpleChatModel, Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.word_count = {}
    
    def __len__(self):
        return len(self.word2idx)
        
    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        
        for word, count in self.word_count.items():
            if count >= 2:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        logger.info(f"词汇表大小: {len(self.word2idx)}")
        
    def encode(self, sentence, max_len):
        indices = []
        for word in sentence.split():
            if len(indices) >= max_len - 1:
                break
            idx = self.word2idx.get(word, self.word2idx['<UNK>'])
            indices.append(idx)
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx.get('<EOS>', 3):
                break
            if idx not in [self.word2idx.get('<BOS>', 2), self.word2idx.get('<PAD>', 0)]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word
            }, f, ensure_ascii=False)
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}


class ChatDataset(Dataset):
    def __init__(self, questions, answers, vocab, max_len):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q_indices = self.vocab.encode(self.questions[idx], self.max_len)
        a_indices = self.vocab.encode(self.answers[idx], self.max_len)
        
        decoder_input = [self.vocab.word2idx['<BOS>']] + a_indices
        decoder_target = a_indices + [self.vocab.word2idx['<EOS>']]
        
        q_indices = self._pad(q_indices, self.max_len)
        decoder_input = self._pad(decoder_input, self.max_len + 1)
        decoder_target = self._pad(decoder_target, self.max_len + 1)
        
        return {
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long)
        }
    
    def _pad(self, seq, max_len):
        if len(seq) < max_len:
            seq += [self.vocab.word2idx['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq


def load_data(file_path):
    questions, answers = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                q, a = parts[0].strip(), parts[1].strip()
                if q and a:
                    questions.append(q)
                    answers.append(a)
    logger.info(f"加载了 {len(questions)} 个对话对")
    return questions, answers


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        
        optimizer.zero_grad()
        output, _ = model(decoder_input)
        
        output = output.view(-1, output.size(-1))
        decoder_target = decoder_target.view(-1)
        loss = criterion(output, decoder_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)
            output, _ = model(decoder_input)
            output = output.view(-1, output.size(-1))
            decoder_target = decoder_target.view(-1)
            loss = criterion(output, decoder_target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    data_file = 'data/train.txt'
    if not os.path.exists(data_file):
        logger.error(f"数据文件 {data_file} 不存在！")
        logger.info("请创建 data/train.txt 文件")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    questions, answers = load_data(data_file)
    if len(questions) == 0:
        logger.error("没有有效的训练数据！")
        return
    
    vocab = Vocabulary()
    vocab.build(questions + answers)
    vocab.save('checkpoints/vocab.json')
    
    config = Config()
    split_idx = int(len(questions) * config.train_ratio)
    
    train_dataset = ChatDataset(questions[:split_idx], answers[:split_idx], vocab, config.max_seq_len)
    val_dataset = ChatDataset(questions[split_idx:], answers[split_idx:], vocab, config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"训练集: {len(train_dataset)} 对, 验证集: {len(val_dataset)} 对")
    
    model = SimpleChatModel(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    
    best_val_loss = float('inf')
    
    logger.info("="*50)
    logger.info("开始训练...")
    logger.info("="*50)
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            logger.info(f"  -> 保存最佳模型")
    
    logger.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()