import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import requests
import base64
from PIL import Image
import io
import numpy as np

import magnum as mn
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image
)

from habitat_sim.nav import ShortestPath
from habitat_sim.nav import PathFinder

import quaternion
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
import os
import re
from collections import defaultdict
import json

from Levenshtein import distance
from gpt_script.qwen_api import process_image_with_prompt
# API endpoint
url = "http://localhost:8000/analyze"

def save_image(epsode_id, trajectory_id, frame_id, img_path, observations):

    image_name = "t" + str(trajectory_id) + "_f" + str(frame_id) + ".png"
    image_full_path = os.path.join(img_path, image_name)
    depth_full_path = os.path.join(img_path, "depth_" + image_name)
    occupancy_map_path = os.path.join(img_path, "occupancy_" + image_name)
    if not cv2.imwrite(image_full_path, transform_rgb_bgr(observations["rgb"])):
        raise ValueError(f"path not exist!!")
    
    # 处理和保存egomap
    egomap = observations["ego_map"]  # ego_map shape: (31, 31, 2)
    
    # 方法1：如果只需要保存单通道二值图
    binary_map = (egomap[:, :, 0] > 0).astype(np.uint8) * 255
    if not cv2.imwrite(occupancy_map_path, binary_map):
        raise ValueError(f"Failed to save occupancy map at {occupancy_map_path}")

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
# response = requests.post(url, json=payload)
# result = response.json()
# print(result["response"])


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def depth_to_world_point3(pixel_x, pixel_y, env):
    """
    Convert a pixel with depth to world coordinates
    
    Args:
        pixel_x: x coordinate of the pixel
        pixel_y: y coordinate of the pixel 
        env: habitat environment instance
    
    Returns:
        world_point: 3D point in world coordinates
    """
    # Get camera parameters
    hfov = 90 * np.pi / 180
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])
    
    # Get depth value
    depth_value = env._sim.get_sensor_observations()['depth'][pixel_y, pixel_x]
    depth_value = depth_value * DEPTH_SHRINK #debug scale 深度图近一点
    # Get camera pose
    camera_state = env.sim.get_agent_state().sensor_states['depth']
    position = camera_state.position
    rotation = camera_state.rotation  # This is a quaternion
    
    width, height = 640 , 480
    # Normalize pixel coordinates to [-1,1]
    x = (pixel_x - width/2) / (width/2)  
    y = -(pixel_y - height/2) / (height/2) # Flip y because image coordinates are y-down

    # Create homogeneous coordinates with depth
    point_camera = np.array([
        x * depth_value,
        y * depth_value,
        -depth_value,  # Negative because camera looks along -Z
        1.0
    ])
    
    # Unproject using inverse of intrinsic matrix
    point_camera = np.matmul(np.linalg.inv(K), point_camera)
    
    # Convert camera pose to transformation matrix
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = quaternion.as_rotation_matrix(rotation)
    T_world_camera[0:3, 3] = position
    
    # Transform point to world coordinates
    point_world = np.matmul(T_world_camera, point_camera)

    return point_world[:3] # Return just x,y,z



def world_to_map(position, top_down_map, sim):
    """
    将世界坐标转换为地图坐标
    注意：Habitat中x-z平面是水平面，y是高度
    """
    map_x, map_y = maps.to_grid(
        position[2],  # z坐标
        position[0],  # x坐标
        top_down_map.shape[0:2],
        sim=sim
    )
    return map_x, map_y

def are_points_in_same_island(pathfinder, point1, point2):
    # 创建 ShortestPath 对象
    path = ShortestPath()
    path.requested_start = point1  # 起点
    path.requested_end = point2   # 终点

    # 检查路径是否可行
    if pathfinder.find_path(path):
        return True  # 存在可行路径，说明在同一个岛屿
    else:
        return False  # 不存在可行路径，说明在不同岛屿
    
img_save_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/full_path_test"

def full_episode_over(position, goal_position):
    if np.linalg.norm(position - goal_position) < 0.5:
        return True
    else: 
        return False

def get_result_from_gpt():
    API_KEY = "sk-KlQFn0xmaR52YmJNwH4hXjLbK6m4kPJo8J8JXtjAI213qgr0" # expensive
    BASE_URL = "https://lonlie.plus7.plus/v1"
    MODEL = "gpt-4o-2024-11-20"

class Molmo:
    def __init__(self, env):
        self.env = env
        self.new_landmark_flag = True
        self.world_next_navigable_point = None
        self.follower = ShortestPathFollower(
            env.sim, goal_radius=0.5, return_one_hot=False
        )
        self.distance = 0
        self.pathfinder = PathFinder()
        self.text = ""
        self.obs = None
        self.text_instruction = ""
        self.history = []
        
    @staticmethod
    def point_move_closer(point_agent, point_target, distance=1):
        """
        让target向agent移动distance距离, numpy
        """
        direction = point_agent - point_target
        direction = direction / np.linalg.norm(direction) * distance
        return point_target + direction


    def get_action(self, obs):
        self.obs = obs
        #先计算距离
        if self.world_next_navigable_point is not None:
            self.distance = abs(np.linalg.norm(self.env.sim.get_agent_state().position - self.world_next_navigable_point))

        if self.distance < 0.5:
            self.new_landmark_flag = True
        
        # 如果到达landmark or 刚刚开始，寻找下一个landmark #maybe stop
        if self.new_landmark_flag:  
            print("----new_landmark_flag == True")
            result_type, result = self.get_xy_or_action_from_llm()
            if result_type == "xy":
                x_llm, y_llm = result
                self.new_landmark_flag = False
            elif result == "arrived":
                return 0
            elif result == "turn right":
                return 3
            elif result == "turn left":
                return 2 
            
            self.world_next_navigable_point = depth_to_world_point3(x_llm, y_llm, self.env) 
            self.world_next_navigable_point[1] = self.env.sim.get_agent_state().position[1] # 把这个点投影到地面
            self.new_landmark_flag = False 
        
        self.snap_world_next_navigable_point = self.env.sim.pathfinder.snap_point(self.world_next_navigable_point)

        while np.isnan(self.snap_world_next_navigable_point)[0]:
            
            #挪过来一点，retry
            self.world_next_navigable_point = self.point_move_closer(self.env.sim.get_agent_state().position, self.world_next_navigable_point)
            self.snap_world_next_navigable_point = self.env.sim.pathfinder.snap_point(self.world_next_navigable_point)
            
        next_action = self.follower.get_next_action(self.snap_world_next_navigable_point)
        
        

        print("----distance to cur_target", abs(np.linalg.norm(self.env.sim.get_agent_state().position - self.world_next_navigable_point)))
        print("----next_action:", next_action)
        return next_action
    
    def get_snap_world_next_navigable_point(self):
        return self.snap_world_next_navigable_point

    def reset(self):
        self.new_landmark_flag = True
        self.world_next_navigable_point = None
        self.snap_world_next_navigable_point = None
        self.distance = 0


    # 'image': ('image.jpg', open('workplace/habitat-lab/AEQA/full_path_test/t42_f16.png', 'rb'), 'image/jpeg'),
    # 'prompt': (None, ' point to the door.')
    def get_xy_or_action_from_llm(self):
        url = "http://localhost:8000/generate"
        obs = self.obs

        instruction = self.env.current_episode.instruction.instruction_text
        history = self.history
        history = ""
        #准备请求文件 file
        rgb_array = obs["rgb"]
        image = Image.fromarray(rgb_array) 
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)  # 确保指针回到字节流的开头 # exmaple: <"the left door">
        prompt = f'You are a robot programmed to navigate by following instructions. Your task is follow the instruction: "{instruction}" . your output history is :"{history}"\
        Please navigate according to the given instructions. If you believe you have arrived near the destination, respond with <arrived>.\
        If you have not yet arrived, identify an object in the scene that you need to approach and respond with object name <object name> \
        If there is no suitable object to approach, you may choose to turn either 90 degrees to the left or 90 degrees to the right. Respond accordingly with <turn left> or <turn right>.\
        Please think step by step, explain your thought process clearly, and then provide your answer in the FORMAT: <answer>. '

        files = {
            'image': ('image.jpg', buffer, 'image/jpeg'),
            'prompt': (None, prompt) #
        }
        cv2.imwrite('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_molmo/test.png', rgb_array)
        # response = requests.post(url, files=files) #这个是出文本 还没到xy
        # result = response.json()['generated_text']
        result = process_image_with_prompt(files=files)
        print("======result======", result)
        import ipdb; ipdb.set_trace()
        
        result_type_keyword, action_word = self.extract_keywords(result)

        self.history.append(action_word)

        result = action_word
        x_flat, y_flat = None, None
        result_type = "action"
        print("======result_type_keyword======", result_type_keyword)
        if result_type_keyword != 'action': # 不是旋转和stop, 提取坐标
            # action name to xy_with_noise
            
            extract_xy_with_noise_result = self.extract_xy_with_noise(action_word) #ask gpt
            x_flat, y_flat = self.extract_coordinates(extract_xy_with_noise_result)
            result_type = "xy"
            result = [x_flat, y_flat]
            print("======goingto======", result)
            if x_flat is None: # 坐标提取不出来
                print("Invalid response from LLaVA:",result["response"])

        return result_type, result
    
    def extract_xy_with_noise(self, text):
        obs = self.obs
        rgb_array = obs["rgb"]
        image = Image.fromarray(rgb_array) 
        cv2.imwrite('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_molmo/test1.png', rgb_array)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)  # 确保指针回到字节流的开头 # exmaple: <"the left door">
        prompt = f'point to {text}'

        files = {
            'image': ('image.jpg', buffer, 'image/jpeg'),
            'prompt': (None, prompt) #
        }
        response = requests.post(url, files=files)
        import ipdb; ipdb.set_trace()
        result = response.json()['generated_text']
        return result
    
    def extract_coordinates(self, text):
        # 定义正则表达式
        pattern = r'<point x="([\d.]+)" y="([\d.]+)"'
        match = re.search(pattern, text)
        x, y = None, None
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
        return x, y
    
    def extract_keywords(self, text): 
        # turn right or turn left or object name or arrived
        # 使用正则表达式匹配 <"turn right"> 格式的内容
        match = re.search(r'<.*?>', text)
        match_list = ['<turn right>', '<turn left>', '<arrived>']
        matched_text = None

        if match:
            matched_text = match.group()  # 返回匹配的内容
            matched_text = matched_text.lower() 

        for _match_word in match_list: # distance match 
            if distance(matched_text, _match_word) < 2:
                matched_text = _match_word
                break
        
        result_type = "action"
        if matched_text == '<"arrived">':
            next_action = "arrived"
        elif matched_text == '<"turn right">':
            next_action = "turn right"
        elif matched_text == '<"turn left">':
            next_action = "turn left"
        else :
            next_action = matched_text
            result_type = "xy"

        return result_type, next_action

PROMPT_TEXT = "describe this image, name a object in the image"
PROMPT_XY = "point to  "

DEPTH_SHRINK = 0.5
END_IDX = 10
img_save_folder = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_topdown"

def example():
    env = habitat.Env(
        config=habitat.get_config("AEQA/vln_r2r_molmo_test.yaml")
    )
    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    print("Environment creation successful")
    cnt = 0
    get_new_landmark = True
    molmo = Molmo(env)

    #benchmark
    episode_predictions = defaultdict(list)
    for _ in range(len(env.episodes[:END_IDX])):
        obs = env.reset()
        frame_step = 0
        print("start a new episode",env.current_episode.episode_id, "cnt:", cnt)
        cnt += 1

        print("Agent stepping around inside environment.")
        history_action = ""
        count_steps = 0
        
        #benchmark
        episode_id = env.current_episode.episode_id
        camera_state = env.sim.get_agent_state().sensor_states['depth']
        episode_predictions[episode_id].append(env.get_info(camera_state))
        while not env.episode_over:
            # flag = save_image(
            #     epsode_id=env.current_episode.episode_id,
            #     trajectory_id=env.current_episode.trajectory_id,
            #     frame_id=frame_step,
            #     img_path=img_save_folder,
            #     observations=obs
            # )

            action = molmo.get_action(obs)
            # 如果转弯 一部只有15度，多转几次
            obs = env.step(action)

            episode_predictions[episode_id].append(env.get_info(camera_state))
            count_steps += 1
            #SAVE IMG
            frame_step += 1
            
            #SAVE TOPDOWN
            # info = env.get_metrics()
            # print("world_to_map", molmo.get_snap_world_next_navigable_point())
            # x1, y1 = world_to_map(molmo.get_snap_world_next_navigable_point(), info["top_down_map"]['map'], env.sim)
            # info["top_down_map"]['pre_agent_map_coord'] = (x1, y1)
            # frame = observations_to_image(obs, info)
            # info.pop("top_down_map")
            # cv2.imwrite('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_topdown/test.png', frame)
            
        molmo.reset()
        print("Episode finished after {} steps.".format(count_steps))
    # Save predictions
    json_path_inference = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debugjson/inference.json"
    with open(json_path_inference, "w") as f:
        json.dump(episode_predictions, f, indent=2)

    print(f"Predictions saved to: {json_path_inference}")

if __name__ == "__main__":
    example()