import re
import json

def sort_key(item):
    # Extract the number from image_name (e.g., "t1192" -> 1192)
    number = int(item['image_name'].split('_')[0][1:])
    return number

def find_missing_qa(json_file):
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {json_file} is not valid JSON")
        return []
    
    # 首先对数据进行排序
    data = sorted(data, key=sort_key)
    
    # 将排序后的数据写回文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # 存储没有question和answer的项
    missing_qa = []
    
    # 遍历所有项
    for item in data:
        if 'question' not in item or 'answer' not in item:
            missing_qa.append(item)
    
    return missing_qa

def main():
    # 使用示例
    json_file = '/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/v2/hand_clean/output_step1_val_seen.json'  # 替换为你的JSON文件路径
    missing_items = find_missing_qa(json_file)
    
    print(f"Found {len(missing_items)} items missing question/answer fields:")
    for item in missing_items:
        print(f"\nImage: {item['image_name']}")
        print(f"Category: {item['category']}")
        print(f"Result: {item['result']}")
        print("-" * 50)

if __name__ == "__main__":
    main()