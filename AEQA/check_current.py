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

img_save_path = "/marcus/habitat-lab/data/collect_data_224_skip2"



def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def save_image(epsode_id, frame_id, img_path, observations):
    image_name = str(epsode_id) + "_" + str(frame_id) + ".png"
    image_full_path = os.path.join(img_path, image_name)
    cv2.imwrite(image_full_path, transform_rgb_bgr(observations["rgb"]))
    return image_full_path

def full_episode_over(position, goal_position):
    if np.linalg.norm(position - goal_position) < 0.5:
        return True
    else:
        return False


def example():
    env = habitat.Env(
        config=habitat.get_config("AEQA/vln_r2r_21.yaml")
    )
    #env.sim.get_agent_state()
    #env.seed(25)
    
    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    for _ in range(len(env.episodes)):
        env.reset()
        folder_path = img_save_path
        file_name = str(env.current_episode.episode_id) + "_0.png"
        folder_path = img_save_path + "/" + file_name
        if not os.path.isfile(folder_path):
            print("====================cant find: ", folder_path)
            return

                

if __name__ == "__main__":
    example()