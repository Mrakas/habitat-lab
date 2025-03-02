import numpy as np

import json


path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/QA_data_val_unseen.json"

new_keys = {'Functional Reasoning', 'Object State Recognition', 'Spatial Reasoning', 'Attribute Recognition', 'Object Localization', 'Object Recognition', 'World Knowledge'}
with open(path, "r") as f:
    data = json.load(f)
cnt = 0
for v in data:
    #import ipdb; ipdb.set_trace()
    # if len(data[v]['QA'].keys()) == 7:
    #     data[v]['QA'] = {new_key: data[v]['QA'][new_key] for new_key in new_keys}
    
    if data[v]['QA'].keys() != new_keys:
        print("v:",v)
        print("data[v]['QA'].keys():",data[v]['QA'].keys())
        cnt += 1

print("cnt:",cnt)

# with open(path, "w") as f:
#     json.dump(data, f, indent=4)
    
#import ipdb; ipdb.set_trace()