import json
import numpy as np
predictions_json_path = "/mnt/data5/ghx/workplace/VLN-CE/predictions_val_unseen.json"
withQA_json_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/val_unseen_with_QA.json"
model_json_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/val_unseen_with_QA_model_gen.json"
"""
1.做成一个新的json文件，将原来的json文件中的QA信息加入到新的json文件中
2.根据 pred 修改起点
"""
with open(predictions_json_path, "r") as f:
    pred_data = json.load(f)

with open(withQA_json_path, "r") as f:
    withQA_data = json.load(f)



def head2rot(heading):
    random_rotation = [
        0,
        -np.sin(heading / 2),
        0,
        -np.cos(heading / 2),
    ]
    return random_rotation
print("len pred_data:",len(pred_data), \
    "len withQA_data:",len(withQA_data['episodes']))
if len(pred_data) != len(withQA_data['episodes']):
    exit(0)
episode_id=-1

for episode in withQA_data['episodes']:
    if episode_id == 1838:
        #import ipdb; ipdb.set_trace()
        print("episode_id:",episode_id)
        print("start_position:",pred_data[str(episode_id)][-1]['position'])

    episode_id = episode["episode_id"]
    withQA_data['episodes'][episode_id -1 ]['start_position'] = pred_data[str(episode_id)][-1]['position'] # 终点坐标
    
    rotation = head2rot(pred_data[str(episode_id)][-1]['heading'])
    withQA_data['episodes'][episode_id -1 ]['start_rotation'] = rotation # 终点朝向
    
with open(model_json_path, "w") as f:
    json.dump(withQA_data, f, indent=4)


