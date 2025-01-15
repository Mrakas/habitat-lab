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

img_save_path = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frame"
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def save_image(epsode_id, trajectory_id, frame_id, img_path, observations):
    image_name = "t" + str(trajectory_id) + "_f" + str(frame_id) + ".png"
    image_full_path = os.path.join(img_path, image_name)
    if not cv2.imwrite(image_full_path, transform_rgb_bgr(observations["rgb"])):
        raise ValueError(f"path not exist!!")
    return image_full_path

def full_episode_over(position, goal_position):
    if np.linalg.norm(position - goal_position) < 0.5:
        return True
    else: 
        return False

SKIP_FRAME = 1 #多少帧采集一次图片
START_IDX = 0
END_IDX = 100 #采集多少张图片

def example():
    env = habitat.Env(
        config=habitat.get_config("AEQA/vln_r2r.yaml"),
        dataset="/mnt/data5/ghx/workplace/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
    )
    #env.sim.get_agent_state()
    #env.seed(25)
    import ipdb;ipdb.set_trace()
    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    cnt = 0
    env.current_episode = env.episodes[0]
    for _ in range(len(env.episodes[START_IDX:END_IDX])):
        env.reset()
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
                    #import ipdb; ipdb.set_trace()
                    #flag = save_image(epsode_id=env.current_episode.episode_id, frame_id=frame_step, img_path=img_save_path, observations=obs)
                    flag = True
                    if flag == False:
                        print("save image failed")
                        return 0
                frame_step += 1
                #env.current_episode
        flag = save_image(epsode_id=env.current_episode.episode_id, \
                          trajectory_id=env.current_episode.trajectory_id,  \
                            frame_id=frame_step, img_path=img_save_path, observations=obs)
        #import ipdb; ipdb.set_trace()

                
if __name__ == "__main__":
    example()
    print("collected:", END_IDX)