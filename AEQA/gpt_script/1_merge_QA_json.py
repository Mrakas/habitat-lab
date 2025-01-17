import json

vln_json_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/see_val_unseen.json"
qa_json_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/QA_data_val_unseen.json"
new_json_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/gpt_script/jsons/val_unseen_with_QA.json"

with open(vln_json_path, "r") as f:
    vln_data = json.load(f)

with open(qa_json_path, "r") as f:
    qa_data = json.load(f)

print("len vln_data:",len(vln_data['episodes']), \
    "len qa_data:",len(qa_data))
trajectory_id=0

for episode in vln_data['episodes']:
    print(trajectory_id)
    # if trajectory_id > 1837:
    #     import ipdb; ipdb.set_trace()

    trajectory_id = episode['trajectory_id']
    QA = qa_data[str(trajectory_id )]['QA']

    for episode in vln_data['episodes']:
        if episode['trajectory_id'] == trajectory_id:
            episode['QA'] = QA

with open(new_json_path, "w") as f:
    json.dump(vln_data, f, indent=4)
print("save to ",new_json_path)

