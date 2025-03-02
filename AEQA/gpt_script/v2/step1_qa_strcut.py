import json
import re

def extract_qa(text):
    # 移除可能的JSON代码块标记
    text = text.replace("```json", "").replace("```", "")
    
    # 尝试解析为JSON格式
    try:
        data = json.loads(text)
        return data["question"], data["answer"]
    except:
        pass
    
    # 所有匹配模式
    patterns = [
        # 带转义引号和换行符的格式
        r'"question":\s*"([^"]+)",[^\n]*\n[^"]*"answer":\s*"([^"]+)"',
        
        # 带引号的格式
        r'"question":\s*"([^"]+)"[^"]+answer":\s*"([^"]+)"',
        
        # 带**的格式（允许换行符）
        r'\*\*Question\*\*:\s*([^\n]+)\n\*\*Answer\*\*:\s*([^\n]+)',
        
        # 带**的格式（包含可能的空格和特殊字符）
        r'\*\*Question:\*\*\s*(.*?)\s*\n\*\*Answer:\*\*\s*(.*?)(?=\n|$)',
        
        # 简单引号格式（无JSON结构）
        r'"question":\s*"([^"]+)"\s*\n"answer":\s*"([^"]+)"',
        
        # 不带引号的格式
        r'question:\s*([^\n]+)\nanswer:\s*([^\n]+)',
        
        # 带问号的格式
        r'([^.\n]+\?)\s*\n\s*([^\n]+)',
        
        # Question/Answer格式（不带特殊标记）
        r'Question:\s*([^\n]+)\s*\nAnswer:\s*([^\n]+)',
        
        # 带连字符的格式
        r'Q-\s*([^\n]+)\s*\nA-\s*([^\n]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    
    return None, None

def process_data(input_data):
    for item in input_data:
        question, answer = extract_qa(item["result"])
        if question and answer:
            item["question"] = question
            item["answer"] = answer
    return input_data

json_path = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/gpt_json/gpt4o_gen_last_frame_val_unseen_rgb_semantic.json"

# 读取JSON文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理数据
processed_data = process_data(data)

# 保存处理后的JSON文件
with open('output_step1_val_unseen.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)