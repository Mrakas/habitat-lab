import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.agent.agent import AgentState, SixDOFPose
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
from scipy.spatial.transform import Rotation

import magnum
from habitat_sim.utils import viz_utils as vut
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image
)
from habitat.utils.visualizations import maps

from habitat_sim.nav import PathFinder
img_save_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_topdown"

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

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

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
    
    
    return image_full_path

def full_episode_over(position, goal_position):
    if np.linalg.norm(position - goal_position) < 0.5:
        return True
    else: 
        return False

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
    depth_value = 0.5 * depth_value #debug scale 深度图近一点
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

SKIP_FRAME = 1 #多少帧采集一次图片
END_IDX = 2 #多少条轨迹？
def example():
    config=habitat.get_config("/mnt/data5/ghx/workplace/habitat-lab/configs/tasks/vln_r2r.yaml")
    print("Task sensors:", config.TASK.SENSORS)
    print("Sensor configs:", config.TASK)
    env = habitat.Env(
        config=habitat.get_config("/mnt/data5/ghx/workplace/habitat-lab/configs/tasks/vln_r2r.yaml")
    )

    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    
    cnt = 0
    env.current_episode = env.episodes[0]
    for _ in range(len(env.episodes[:END_IDX])):
        import ipdb; ipdb.set_trace()
        env.reset()
        path = env.current_episode.reference_path + [
            env.current_episode.goals[0].position
        ]
        frame_step = 0
        print("start a new episode",env.current_episode.episode_id, "cnt:", cnt)
        cnt += 1
        video_path = f"/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_topdown/e{env.current_episode.episode_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        for point in path[1:]:
            while full_episode_over(env.sim.get_agent_state().position, path[-1]) == False:
                best_action = follower.get_next_action(point)
                if best_action == 0:
                    break

                obs = env.step(best_action)
                #深度图点对应的世界坐标

                vis_frames = []
                if frame_step % SKIP_FRAME == 0:
                    
                    # 更新占用栅格地图
                    info = env.get_metrics()
                    #投影到最近的navigable point
                    world_next = depth_to_world_point3(320, 260, env)
                    #world_next = 0.5 * world_next
                    projected_point = magnum.Vector3(world_next[0], world_next[1], world_next[2])
                    world_next_navigable = env.sim.pathfinder.snap_point(projected_point)
                    
                    x1, y1 = world_to_map(world_next, info["top_down_map"]['map'], env.sim)

                    info["top_down_map"]['pre_agent_map_coord'] = (x1, y1)


                    env.sim.get_agent_state().position
                    
                    frame = observations_to_image(obs, info)
                    info.pop("top_down_map")

                    if out is None:
                        height, width = frame.shape[:2]
                        out = cv2.VideoWriter(
                            video_path,
                            fourcc, 
                            1.0,  # fps
                            (width, height)
                        )

                    # 写入视频帧
                    #import ipdb; ipdb.set_trace()
                    out.write(frame)
                    vis_frames.append(frame)
                    cv2.imwrite('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_topdown/test.png', frame)
                    flag = save_image(
                        epsode_id=env.current_episode.episode_id,
                        trajectory_id=env.current_episode.trajectory_id,
                        frame_id=frame_step,
                        img_path=img_save_path,
                        observations=obs
                    )
                    # 每隔一定帧数保存一次地图
                frame_step += 1
        if out is not None:
            out.release()

        # 每个episode结束时保存一次完整地图

if __name__ == "__main__":
    example()
    print("collected:", END_IDX)