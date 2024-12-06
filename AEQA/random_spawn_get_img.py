import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.agent.agent import AgentState, SixDOFPose
import cv2
import os
import json
import uuid
import quaternion
import numpy as np

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"
SAVE_KEY = "k"

img_save_path = "/home/marcus/workplace/habitat-lab/AEQA/data/imgs/"
json_save_path = "/home/marcus/workplace/habitat-lab/AEQA/data/destinations/"

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def save_image_and_location(observations, env, img_path, json_path):
    image_uuid = str(uuid.uuid4())
    image_name = image_uuid + ".png"
    image_full_path = os.path.join(img_path, image_name)
    cv2.imwrite(image_full_path, transform_rgb_bgr(observations["rgb"]))

    agent_state = env.sim.get_agent_state()

    agent_position = agent_state.position.tolist()  # Convert to list for JSON serialization
    agent_rotation = quaternion.as_float_array(agent_state.rotation).tolist()  # Quaternion

    rgb_position = agent_state.sensor_states["rgb"].position.tolist()
    rgb_rotation = quaternion.as_float_array(agent_state.sensor_states["rgb"].rotation).tolist()
    
    depth_position = agent_state.sensor_states["depth"].position.tolist()
    depth_rotation = quaternion.as_float_array(agent_state.sensor_states["depth"].rotation).tolist()

    location_data = {
        "image_uuid": image_uuid,
        "AgentState": {
            "location": {
                "destination_distance": float(observations["pointgoal_with_gps_compass"][0]),#useless
                "theta_radians": float(observations["pointgoal_with_gps_compass"][1])#useless
            },
            "agent_state": {
                "position": agent_position,
                "rotation": agent_rotation
            }
        },
        "sensor_states": {
            "rgb": {
                "position": rgb_position,
                "rotation": rgb_rotation
            },
            "depth": {
                "position": depth_position,
                "rotation": depth_rotation
            }
        }
    }
    
    json_full_path = os.path.join(json_path, image_uuid + ".json")
    with open(json_full_path, 'w') as json_file:
        json.dump(location_data, json_file)

#AgentState(position=array([-7.3699493 ,  0.08276175,  6.5762997 ], dtype=float32), rotation=quaternion(-0.269477784633636, 0, 0.963006675243378, 0), sensor_states={'rgb': SixDOFPose(position=array([-7.3699493,  1.3327618,  6.5762997], dtype=float32), rotation=quaternion(-0.269477784633636, 0, 0.963006675243378, 0)), 'depth': SixDOFPose(position=array([-7.3699493,  1.3327618,  6.5762997], dtype=float32), rotation=quaternion(-0.269477784633636, 0, 0.963006675243378, 0))})  

def from_json_to_state(json_path: str) -> AgentState:
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    position = np.array(data['position'])
    rotation = quaternion.quaternion(*data['rotation'])
    
    sensor_states = {
        sensor: SixDOFPose(
            position=np.array(sensor_data['position']),
            rotation=quaternion.quaternion(*sensor_data['rotation'])
        )
        for sensor, sensor_data in data['sensor_states'].items()
    }
    
    return AgentState(position=position, rotation=rotation, sensor_states=sensor_states)




def example():
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )
    env.sim.get_agent_state
    print("Environment creation successful")
    env.seed(25)
    observations = env.reset()
    #random spawn
    #import pdb; pdb.set_trace()
    random_start_point = env.sim.sample_navigable_point()
    print("Random start point: ", random_start_point)
    #import pdb; pdb.set_trace()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    #cv2.resizeWindow("RGB", 800, 600)

    env.current_episode #curren episode is a dictionary with keys: "scene_id", "start_position", "start_rotation", "info"
    print("Agent stepping around inside environment.")
    
    env.action_space
    env.sim.get_agent_state()

    env.sim.set_agent_state(random_start_point, rotation=quaternion.quaternion(1, 0, 0, 0))
    #import pdb; pdb.set_trace()

    count_steps = 0
    while not env.episode_over:
        #cv2.resizeWindow("RGB", 1800, 600)
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        elif keystroke == ord(SAVE_KEY):
            save_image_and_location(observations, env, img_save_path, json_save_path)
            print("Image and location saved.")
            continue
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        resized_image = cv2.resize(observations["rgb"], (1024, 1024))
        cv2.imshow("RGB", transform_rgb_bgr(resized_image))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()