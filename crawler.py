import requests
import json
import time
import random
import os
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

class LargeScaleCrawler:
    """大规模对话数据爬虫 - 目标10000+条"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.dialogues = []
        
    def generate_massive_data(self):
        """生成大规模模拟数据（实际爬虫的替代方案）"""
        
        # 1. 问候类对话 (500条变体)
        greetings = {
            "你好": ["你好", "您好", "嗨", "哈喽", "hi", "hello", "在吗", "喂", "有人吗", "嘿"],
            "回答": ["你好呀", "您好啊", "嗨～", "哈喽～", "在呢", "来了来了", "到！", "嗯？", "请说", "我在"]
        }
        
        # 2. 日常对话模板 (2000条组合)
        topics = {
            "天气": ["今天天气怎么样", "明天会下雨吗", "热死了", "好冷啊", "天气真好", "刮风了", "下雪了吗"],
            "回答天气": ["今天晴天", "预报说有雨", "开空调吧", "多穿点", "适合出门", "注意安全", "好漂亮"],
            "心情": ["心情不好", "好开心", "郁闷", "烦躁", "兴奋", "紧张", "平静"],
            "回答心情": ["怎么了", "为什么呀", "我陪你", "放松点", "太好了", "别担心", "那就好"],
            "工作": ["加班好累", "工作顺利吗", "要开会了", "项目做完了", "压力大", "辞职了", "升职了"],
            "回答工作": ["辛苦了", "恭喜", "加油", "休息下", "太棒了", "别太累", "为你高兴"],
            "学习": ["考试好难", "作业太多了", "想考研", "学Python", "背单词", "上课困", "毕业了"],
            "回答学习": ["坚持住", "慢慢来", "支持你", "刚开始都难", "一起学", "喝咖啡", "恭喜毕业"],
            "生活": ["吃饭了吗", "睡了没", "在干嘛", "去哪了", "买什么了", "看什么", "听什么"],
            "回答生活": ["吃了", "还没", "看电视", "逛超市", "买衣服", "看电影", "听音乐"],
        }
        
        # 3. 问答类对话 (3000条)
        qa_pairs = [
            ("Python怎么学", "从基础语法开始，每天写代码并做小项目"),
            ("Java和Python哪个好", "各有优势，Python简单，Java稳定"),
            ("怎么找工作", "先确定方向，准备简历，多投递，准备面试"),
            ("如何涨薪", "提升技能，主动承担，和老板沟通"),
            ("怎么减肥", "控制饮食，坚持运动，保持作息规律"),
            ("怎么学英语", "每天背单词，多听多说，看美剧"),
            ("如何早睡", "远离手机，定时作息，白天多运动"),
            ("怎么提高效率", "番茄工作法，优先级排序，减少干扰"),
            ("如何存钱", "记账，定存，减少不必要的消费"),
            ("怎么学做饭", "从简单菜开始，看视频教程，多做实践"),
            ("如何谈恋爱", "真诚沟通，互相理解，保持独立"),
            ("怎么选专业", "兴趣优先，考虑就业，咨询前辈"),
            ("如何克服拖延", "任务拆分，设置deadline，奖励机制"),
            ("怎么提高自信", "小目标达成，正向暗示，接纳自己"),
            ("如何处理人际关系", "换位思考，真诚待人，保持边界"),
            ("怎么选工作", "看发展前景，薪资待遇，工作环境"),
            ("如何买房", "存首付，选地段，算贷款"),
            ("怎么理财投资", "先学习，从小额开始，分散投资"),
            ("如何提升自己", "每天学习，实践反思，交优秀朋友"),
            ("怎么养猫", "准备猫粮猫砂，定期疫苗，多陪伴"),
            ("如何学吉他", "从和弦开始，每天练习，找老师"),
            ("怎么拍好照片", "学习构图，多拍多练，找参考"),
            ("如何写简历", "突出亮点，量化成果，针对性修改"),
            ("怎么面试成功", "准备充分，了解公司，展现自信"),
            ("如何创业", "找准方向，小规模试错，坚持"),
        ]
        
        # 4. 情感类对话 (1500条)
        emotions = {
            "开心": ["太棒了", "好开心", "耶", "高兴", "爽", "完美", "厉害了"],
            "伤心": ["难过", "想哭", "失望", "痛苦", "崩溃", "绝望", "委屈"],
            "愤怒": ["气死了", "烦死了", "受不了", "无语", "爆炸", "抓狂", "火大"],
            "焦虑": ["好慌", "紧张", "担心", "害怕", "不安", "焦虑", "忐忑"],
        }
        
        # 5. 电影台词扩展 (1000条)
        movies = [
            ("活着是为了什么", "为了那些我们还没见过的美好"),
            ("什么是爱情", "爱情是互相理解和包容"),
            ("人生意义是什么", "每个人都有自己的答案"),
            ("怎么才算成功", "成为自己想成为的人"),
            ("友情重要吗", "非常重要，朋友是人生的财富"),
            ("如何面对失败", "失败是成功之母，从中学习"),
            ("梦想能实现吗", "坚持就有可能"),
            ("什么是幸福", "幸福就在当下"),
        ]
        
        print("开始生成大规模数据...")
        
        # 组合生成
        total = 0
        
        # 问候类 (500条)
        for i in range(50):
            for g in greetings["你好"]:
                for a in greetings["回答"]:
                    self.dialogues.append({"question": g, "answer": a})
                    total += 1
                    if total >= 500:
                        break
                if total >= 500:
                    break
            if total >= 500:
                break
        
        # 日常对话 (2000条)
        for _ in range(2000):
            topic_type = random.choice(list(topics.keys()))
            if topic_type.startswith("回答"):
                continue
            question = random.choice(topics[topic_type])
            answer_type = "回答" + topic_type
            if answer_type in topics:
                answer = random.choice(topics[answer_type])
            else:
                answer = "嗯，我明白了"
            self.dialogues.append({"question": question, "answer": answer})
        
        # 问答对 (3000条)
        for _ in range(3000):
            qa = random.choice(qa_pairs)
            self.dialogues.append({"question": qa[0], "answer": qa[1]})
        
        # 情感类 (1500条)
        for emotion, words in emotions.items():
            for word in words:
                for _ in range(50):
                    self.dialogues.append({"question": word, "answer": f"是的，{emotion}就好"})
        
        # 扩展变体 (3000条)
        new_dialogues = []
        for item in self.dialogues[:3000]:
            # 添加标点变体
            new_dialogues.append({
                "question": item["question"] + random.choice(["！", "？", "~", "..."]) if random.random() > 0.5 else item["question"],
                "answer": item["answer"] + random.choice(["！", "~", "哦", "呢", "哈"]) if random.random() > 0.5 else item["answer"]
            })
        
        self.dialogues.extend(new_dialogues)
        
        print(f"生成了 {len(self.dialogues)} 条数据")
        return self.dialogues
    
    def crawl_from_github(self):
        """从GitHub开源数据集爬取"""
        print("尝试从GitHub获取公开数据集...")
        
        # 中文对话数据集地址
        repos = [
            "https://raw.githubusercontent.com/codemayq/chinese_chatbot_corpus/master/data/clean_chat_corpus.tsv",
            "https://raw.githubusercontent.com/Samurais/chatbot-corpus/master/chatbot.csv",
        ]
        
        for url in repos:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    lines = response.text.split('\n')
                    for line in lines:
                        if '\t' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                self.dialogues.append({
                                    "question": parts[0][:50],
                                    "answer": parts[1][:50]
                                })
                    print(f"从 {url} 获取了数据")
            except:
                continue
        
        return self.dialogues
    
    def data_augmentation(self, multiplier=5):
        """数据增强 - 生成5倍数据"""
        print(f"正在进行数据增强 (x{multiplier})...")
        
        augmented = []
        
        # 同义词映射
        synonyms = {
            "你好": ["您好", "嗨", "哈喽", "hi", "你好呀", "你好啊", "你好哦"],
            "谢谢": ["多谢", "感谢", "谢谢啦", "谢了", "感激", "Thanks"],
            "再见": ["拜拜", "下次见", "后会有期", "回见", "再见啦", "see you"],
            "好": ["不错", "很好", "挺好的", "棒", "赞", "优秀", "可以的"],
            "开心": ["高兴", "快乐", "愉悦", "愉快", "美滋滋", "乐呵呵"],
            "漂亮": ["美丽", "好看", "靓丽", "美", "惊艳", "迷人"],
            "聪明": ["机智", "聪慧", "有才", "灵光", "睿智", "天才"],
            "厉害": ["牛", "强", "了不起", "棒", "给力", "牛掰"],
            "喜欢": ["喜爱", "钟意", "欣赏", "中意", "迷上"],
            "知道": ["了解", "明白", "清楚", "晓得", "理解"],
        }
        
        # 语气词
        particles = ["啊", "哦", "呢", "吧", "嘛", "啦", "哟", "哈", "嘿", "哇"]
        
        # 表情符号
        emoticons = ["😊", "😄", "😂", "😭", "😍", "😎", "🥰", "🤔", "👍", "❤️"]
        
        for item in self.dialogues:
            augmented.append(item.copy())
            
            for _ in range(multiplier - 1):
                new_q = item["question"]
                new_a = item["answer"]
                
                # 同义词替换
                for word, syns in synonyms.items():
                    if word in new_q and random.random() > 0.7:
                        new_q = new_q.replace(word, random.choice(syns))
                    if word in new_a and random.random() > 0.7:
                        new_a = new_a.replace(word, random.choice(syns))
                
                # 添加语气词
                if random.random() > 0.8:
                    new_q += random.choice(particles)
                    new_a += random.choice(particles)
                
                # 添加表情
                if random.random() > 0.85:
                    new_q += random.choice(emoticons)
                    new_a += random.choice(emoticons)
                
                # 语句转换
                if random.random() > 0.9:
                    new_q = "请问，" + new_q
                
                augmented.append({"question": new_q[:100], "answer": new_a[:100]})
        
        self.dialogues = augmented
        print(f"增强后共 {len(self.dialogues)} 条")
    
    def remove_duplicates(self):
        """去重"""
        print("正在去重...")
        unique = {}
        for item in self.dialogues:
            key = f"{item['question']}|{item['answer']}"
            if key not in unique:
                unique[key] = item
        
        self.dialogues = list(unique.values())
        print(f"去重后共 {len(self.dialogues)} 条")
    
    def save_to_file(self, filename='data/train.txt'):
        """保存到文件"""
        os.makedirs('data', exist_ok=True)
        
        # 打乱顺序
        random.shuffle(self.dialogues)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.dialogues:
                f.write(f"{item['question']}\t{item['answer']}\n")
        
        print(f"✅ 已保存 {len(self.dialogues)} 条对话到 {filename}")
        
        # 统计信息
        avg_q_len = sum(len(d['question']) for d in self.dialogues) / len(self.dialogues)
        avg_a_len = sum(len(d['answer']) for d in self.dialogues) / len(self.dialogues)
        
        print(f"\n📊 数据统计:")
        print(f"  - 总对话数: {len(self.dialogues):,}")
        print(f"  - 平均问题长度: {avg_q_len:.1f} 字")
        print(f"  - 平均回答长度: {avg_a_len:.1f} 字")
    
    def run(self, target_count=10000):
        """运行完整流程"""
        print("="*60)
        print("🚀 大规模对话数据生成器")
        print(f"🎯 目标: {target_count:,} 条对话")
        print("="*60)
        
        # 生成数据
        self.generate_massive_data()
        
        # 从GitHub获取（如果有网络）
        self.crawl_from_github()
        
        # 数据增强
        current_count = len(self.dialogues)
        if current_count < target_count:
            multiplier = max(2, target_count // current_count)
            self.data_augmentation(multiplier=min(multiplier, 10))
        
        # 去重
        self.remove_duplicates()
        
        # 截取目标数量
        if len(self.dialogues) > target_count:
            self.dialogues = self.dialogues[:target_count]
        
        # 保存
        self.save_to_file()
        
        # 显示样例
        print("\n📝 数据样例:")
        for i in range(min(10, len(self.dialogues))):
            print(f"{i+1}. 问: {self.dialogues[i]['question'][:50]}")
            print(f"   答: {self.dialogues[i]['answer'][:50]}\n")
        
        print("="*60)
        print("✅ 完成！现在可以运行 python train.py 开始训练")
        print("="*60)


if __name__ == "__main__":
    crawler = LargeScaleCrawler()
    
    # 生成100000条数据
    crawler.run(target_count=100000)
