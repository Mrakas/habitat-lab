import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import requests
import base64
from PIL import Image
import io
import numpy as np
# API endpoint
url = "http://localhost:8000/analyze"


def encode_image(image):
    if isinstance(image, np.ndarray):
        # 将 NumPy 数组转换为 PIL 图像
        pil_img = Image.fromarray(image)
        # 将 PIL 图像转换为字节
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        # 将字节编码为 base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif isinstance(image, str):
        # 如果是文件路径，读取图像并编码
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the request
image_path = "/mnt/data5/ghx/workplace/LLaVA-NeXT/test.png"
base64_image = encode_image(image_path)

# Prepare the payload
# payload = {
#     "image": base64_image, # 5张
#     "question": '<img> Please navigate according to the instruction, instruction "xx", please output one of "go forward, left turn, right turn, stop", when the destination is reached output "stop". Your history command is "BBBBB" with no additional output.'
# }

payload = {
    "image": base64_image, # 5张
    "question": 'please describe this image:'
}

# Make the request
response = requests.post(url, json=payload)
result = response.json()
print(result["response"])


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_action_from_llm(img, history):
    base64_image = encode_image(img)
    payload = {
        "image": base64_image,
        "question": f' Please navigate according to the instruction, instruction "{history}", please output one of "move forward, turn left, turn right, stop", and explain your choice thoroughly, when the destination is reached output "stop". Your history command is "{history}" with no additional output.'
    }
    response = requests.post(url, json=payload)
    result = response.json()
    if result["response"] not in ['move forward', 'turn left', 'turn right', 'stop']:
        print("Invalid response from LLaVA:",result["response"])
    return result["response"]

def example():
    
    env = habitat.Env(
        config=habitat.get_config("AEQA/vln_r2r_test.yaml")
    )
    
    print("Environment creation successful")
    observations = env.reset()

    #cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    history_action = ""
    count_steps = 0
    while not env.episode_over:
        keystroke = get_action_from_llm(observations["rgb"], history_action)
        #import ipdb; ipdb.set_trace()
        if keystroke == "move forward":
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == "turn left":
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == "turn right":
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == "stop":
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1


        #save img
        img = Image.fromarray(transform_rgb_bgr(observations["rgb"]))
        save_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug"
        img.save(f"{save_path}/t{count_steps}.png")

        
    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()