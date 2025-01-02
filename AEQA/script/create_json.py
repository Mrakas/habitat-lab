from pathlib import Path
import json

# 指定文件夹路径
folder_path = Path('/home/marcus/workplace/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/train')

# 获取文件夹下所有的子文件夹名字
subfolders = []
for f in folder_path.iterdir():
    if f.is_dir():
        subfolders.append(f.name)


subfolders = sorted(subfolders)
# 将子文件夹名称保存为 JSON 文件
output_file = '/home/marcus/workplace/habitat-lab/AEQA/config/map_list.json'
with open(output_file, 'w') as json_file:
    
    json.dump(subfolders, json_file, indent=4)

print(f"Subfolder names have been saved to {output_file}")
