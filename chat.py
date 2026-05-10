import torch
import json
import os
import sys
from model import SimpleChatModel, Config

class ChatBot:
    """对话机器人交互类"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', vocab_path='checkpoints/vocab.json'):
        """
        初始化聊天机器人
        Args:
            model_path: 模型权重文件路径
            vocab_path: 词汇表文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 检查文件是否存在
        if not os.path.exists(vocab_path):
            print(f"错误: 词汇表文件 {vocab_path} 不存在！")
            print("请先运行 train.py 训练模型")
            sys.exit(1)
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在！")
            print("请先运行 train.py 训练模型")
            sys.exit(1)
        
        # 加载词汇表
        self.vocab = self.load_vocab(vocab_path)
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        print(f"模型加载成功！词汇表大小: {len(self.vocab)}")
        
    def load_vocab(self, vocab_path):
        """加载词汇表"""
        class Vocabulary:
            def __init__(self):
                self.word2idx = {}
                self.idx2word = {}
            
            def __len__(self):
                return len(self.word2idx)
            
            def decode(self, indices):
                """索引转句子"""
                words = []
                eos_id = self.word2idx.get('<EOS>', 3)
                bos_id = self.word2idx.get('<BOS>', 2)
                pad_id = self.word2idx.get('<PAD>', 0)
                
                for idx in indices:
                    if idx == eos_id:
                        break
                    if idx not in [bos_id, pad_id]:
                        words.append(self.idx2word.get(idx, '<UNK>'))
                return ' '.join(words)
            
            def encode(self, sentence, max_len=15):
                """句子转索引"""
                indices = []
                unk_id = self.word2idx.get('<UNK>', 1)
                for word in sentence.split():
                    if len(indices) >= max_len - 1:
                        break
                    idx = self.word2idx.get(word, unk_id)
                    indices.append(idx)
                return indices
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = Vocabulary()
        vocab.word2idx = vocab_data['word2idx']
        # 确保 idx2word 的键是整数
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
        return vocab
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取词汇表大小
        vocab_size = len(self.vocab)
        
        # 创建模型
        config = Config()
        model = SimpleChatModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint)
        model.eval()
        
        return model
    
    def get_response(self, question, max_length=20, temperature=0.7, top_k=50):
        """
        获取机器人回复
        Args:
            question: 用户输入的问题
            max_length: 最大回复长度
            temperature: 温度参数 (0.1-1.0)，越低越确定
            top_k: top-k采样
        Returns:
            回复文本
        """
        if not question.strip():
            return "请说点什么吧..."
        
        # 编码问题
        input_indices = self.vocab.encode(question, max_length)
        
        if len(input_indices) == 0:
            return "我没听清楚，能再说一遍吗？"
        
        # 转换为tensor
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor, 
                max_length=max_length, 
                temperature=temperature,
                top_k=top_k
            )
        
        # 解码回复
        response = self.vocab.decode(generated_ids)
        
        return response if response else "嗯...我在思考中"
    
    def interactive_mode(self):
        """交互式对话模式"""
        print("\n" + "="*50)
        print("🤖 对话AI已启动")
        print("="*50)
        print("命令说明:")
        print("  - 直接输入问题开始对话")
        print("  - 'quit' 或 'exit' 退出程序")
        print("  - 'clear' 清屏")
        print("  - 'temp [值]' 调整温度参数 (当前: 0.7)")
        print("  - 'topk [值]' 调整top-k参数 (当前: 50)")
        print("="*50)
        print("开始对话吧！\n")
        
        temperature = 0.7
        top_k = 50
        
        while True:
            try:
                # 获取用户输入
                user_input = input("你: ").strip()
                
                # 处理命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！👋")
                    break
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                elif user_input.lower().startswith('temp'):
                    try:
                        parts = user_input.split()
                        if len(parts) > 1:
                            new_temp = float(parts[1])
                            if 0.1 <= new_temp <= 2.0:
                                temperature = new_temp
                                print(f"温度参数已调整为: {temperature}")
                            else:
                                print("温度值应在 0.1-2.0 之间")
                        else:
                            print(f"当前温度参数: {temperature}")
                    except ValueError:
                        print(f"当前温度参数: {temperature}")
                    continue
                
                elif user_input.lower().startswith('topk'):
                    try:
                        parts = user_input.split()
                        if len(parts) > 1:
                            new_topk = int(parts[1])
                            if 1 <= new_topk <= 100:
                                top_k = new_topk
                                print(f"Top-K参数已调整为: {top_k}")
                            else:
                                print("Top-K值应在 1-100 之间")
                        else:
                            print(f"当前Top-K参数: {top_k}")
                    except ValueError:
                        print(f"当前Top-K参数: {top_k}")
                    continue
                
                elif not user_input:
                    continue
                
                # 获取回复
                response = self.get_response(user_input, temperature=temperature, top_k=top_k)
                print(f"AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n再见！👋")
                break
            except Exception as e:
                print(f"错误: {e}")
                continue
    
    def single_test(self):
        """单次测试模式"""
        print("\n" + "="*50)
        print("🤖 单次测试模式")
        print("="*50)
        print("输入问题后按回车，AI会回复一次")
        print("输入 'quit' 退出\n")
        
        while True:
            question = input("问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not question:
                continue
            
            response = self.get_response(question)
            print(f"回答: {response}\n")
    
    def batch_test(self, test_questions):
        """批量测试模式"""
        print("\n" + "="*50)
        print("🤖 批量测试模式")
        print("="*50)
        
        for i, question in enumerate(test_questions, 1):
            response = self.get_response(question)
            print(f"{i}. 问: {question}")
            print(f"   答: {response}\n")
        
        print("批量测试完成！")


def show_menu():
    """显示主菜单"""
    print("\n" + "="*50)
    print("        对话AI - 启动菜单")
    print("="*50)
    print("1. 交互式对话模式")
    print("2. 单次测试模式")
    print("3. 批量测试模式")
    print("4. 退出")
    print("="*50)


def main():
    """主函数"""
    # 检查必要文件
    if not os.path.exists('checkpoints/best_model.pth'):
        print("错误: 未找到训练好的模型！")
        print("请先运行 train.py 进行训练")
        return
    
    if not os.path.exists('checkpoints/vocab.json'):
        print("错误: 未找到词汇表文件！")
        print("请先运行 train.py 进行训练")
        return
    
    # 创建聊天机器人
    try:
        bot = ChatBot()
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 预设测试问题（用于批量测试）
    test_questions = [
        "你好",
        "你叫什么名字",
        "今天天气怎么样",
        "你能做什么",
        "再见"
    ]
    
    # 主菜单循环
    while True:
        show_menu()
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            bot.interactive_mode()
        elif choice == '2':
            bot.single_test()
        elif choice == '3':
            bot.batch_test(test_questions)
        elif choice == '4':
            print("感谢使用，再见！")
            break
        else:
            print("无效选择，请重新输入")
        
        # 返回菜单前暂停
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()