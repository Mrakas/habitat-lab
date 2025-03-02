import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import habitat.tasks
import habitat.tasks.nav
import habitat.tasks.nav.nav
from habitat_sim.agent.agent import AgentState, SixDOFPose
import habitat_sim
import cv2
import os
import json
import uuid
import quaternion
import numpy as np
import random
import time
import sys
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from PIL import Image
from tqdm import tqdm

img_save_path = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/last_frame_test_rgb_semantic"
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def save_image(epsode_id, trajectory_id, frame_id, img_path, observations, save_semantic=True):
    image_name = "t" + str(trajectory_id) + "_f" + str(frame_id) + ".png"
    image_full_path = os.path.join(img_path, image_name)
    sem_full_path = os.path.join(img_path, "semantic_" + image_name)
    #import ipdb;ipdb.set_trace()
    if save_semantic:
        sem = observations["semantic"]  # 256*256 uint32
        # 将uint32拆分为RGB通道
        r = (sem >> 16) & 255  # 取高8位作为R通道
        g = (sem >> 8) & 255   # 取中间8位作为G通道
        b = sem & 255          # 取低8位作为B通道
        
        # 堆叠RGB通道
        sem_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        
        # 保存语义分割图像
        if not cv2.imwrite(sem_full_path, transform_rgb_bgr(sem_rgb)):
            raise ValueError(f"Failed to save semantic image at {sem_full_path}")

    if not cv2.imwrite(image_full_path, transform_rgb_bgr(observations["rgb"])): # observations["semantic"]
        raise ValueError(f"path not exist!!")
    return image_full_path

def full_episode_over(position, goal_position):
    if np.linalg.norm(position - goal_position) < 0.5:
        return True
    else: 
        return False

SKIP_FRAME = 1 #多少帧采集一次图片
START_IDX = 0
END_IDX = 99999 #采集多少张图片

def example():
    config=habitat.get_config("/mnt/data5/ghx/workplace/habitat-lab/configs/tasks/vln_r2r.yaml")
    env = habitat.Env(
        config
    )
    #config=habitat.get_config("/mnt/data5/ghx/workplace/habitat-lab/configs/tasks/vln_r2r.yaml")
    env.sim.get_agent_state()
    #env.sim.get_agent_state()
    #env.seed(25)

    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    cnt = 0
    env.current_episode = env.episodes[0]
    print(f"save path: {img_save_path} \n \
          total num: {len(env.episodes[START_IDX:END_IDX])}")

    for _ in tqdm(range(len(env.episodes[START_IDX:END_IDX]))): #tqdm
        obs1 = env.reset()
        import ipdb;ipdb.set_trace()
        path = env.current_episode.reference_path + [
            env.current_episode.goals[0].position
        ]
        frame_step = 0
        print("start a new episode",env.current_episode.episode_id, "cnt:", cnt)
        cnt += 1
        for point in path[1:]:#第一个point 和起点一样 skip
            
            while full_episode_over(env.sim.get_agent_state().position, path[-1]) == False:
                
                best_action = follower.get_next_action(point)
                if best_action == 0:
                    break

                obs = env.step(best_action)

                #print("action:", best_action, "frame_step:", frame_step)
                #动两次保存一次图片
                if frame_step % SKIP_FRAME == 0:
                    flag = True
                    if flag == False:
                        print("save image failed")
                        return 0
                frame_step += 1
                #env.current_episode

        flag = save_image(epsode_id=env.current_episode.episode_id, \
                          trajectory_id=env.current_episode.trajectory_id,  \
                            frame_id=frame_step, img_path=img_save_path, observations=obs, \
                                save_semantic=True)
        if flag == False:
            print("save image failed")
            return 0
        
        #import ipdb;ipdb.set_trace()


def point_in_bbox(point, bbox):
    """检查点是否在包围盒内"""
    center = bbox.center
    sizes = bbox.sizes
    
    # 计算包围盒的最小和最大点
    half_sizes = sizes / 2
    min_point = center - half_sizes
    max_point = center + half_sizes
    
    return (point[0] >= min_point[0] and point[0] <= max_point[0] and
            point[1] >= min_point[1] and point[1] <= max_point[1] and
            point[2] >= min_point[2] and point[2] <= max_point[2])

def get_current_room_info(env):
    """获取当前位置的房间信息"""
    agent_state = env.sim.get_agent_state()
    position = agent_state.position
    semantic_scene = env.sim.semantic_annotations()
    
    room_info = []
    for level in semantic_scene.levels:
        for region in level.regions:
            if point_in_bbox(position, region.aabb):
                room_info.append({
                    'level_id': level.id,
                    'region_id': region.id,
                    'category': region.category.name(),
                    'center': region.aabb.center,
                    'sizes': region.aabb.sizes
                })
    
    return position, room_info

if __name__ == "__main__":
    example()
    print("collected:", END_IDX)