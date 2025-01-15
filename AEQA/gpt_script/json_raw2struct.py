import re
import json
'''
脚本得到的gpt4 response 还没有进行格式化，需要根据规则匹配对脚本进行清洗。
'''
json_path = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/json/data.json"

with open(json_path, "r") as f:
    data = json.load(f)

episode_dict = {}

# data['image_name'] , data['result']
for i, cur_data in enumerate(data):
    appenddata = {}
    json_procesing = re.search(r'```json\n(.*?)\n```', cur_data['result'], re.DOTALL)
    json_procesing = json_procesing.group(1)
    json_procesing = json_procesing.replace("\n", "").replace("\\", "")
    json_procesing = json.loads(json_procesing)

    episode_id = cur_data['image_name'].split("_")[0].replace("t", "")

    appenddata['episode_id'] = episode_id
    appenddata['QA'] = json_procesing

    episode_dict[episode_id] = {
        'episode_id': episode_id,
        'QA': json_procesing
    }

    #import ipdb; ipdb.set_trace()

save_path = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/json/processed_data.json"
with open(save_path, "w") as f:
    json.dump(episode_dict, f)

import ipdb; ipdb.set_trace()
print(f"Data successfully saved to {save_path}")