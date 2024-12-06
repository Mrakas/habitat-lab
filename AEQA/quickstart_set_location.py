import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import json
import quaternion

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
RANDOM_MOVE="y"
TMP="p"



def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    
    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )
    
    #read js
    file_path = '/home/marcus/workplace/habitat-lab/AEQA/data/destinations/2912802b-bfe0-421c-9699-5779f93c6897.json'

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(data)

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    
    env.sim.get_agent_state()
    #import ipdb; ipdb.set_trace()
    env.sim.set_agent_state(data['AgentState']['agent_state']['position'], rotation=quaternion.quaternion(*data['AgentState']['agent_state']['rotation']))
    #data['AgentState']['agent_state']['position']

    
    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)
        #import ipdb; ipdb.set_trace()
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
        elif keystroke == ord(RANDOM_MOVE):
            observations = env.reset()
            print("Agent's position has been reset to a random valid point.")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

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