import json
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import habitat_sim
from typing import TYPE_CHECKING, Union, cast
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.agent import Agent
import quaternion
if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
import numpy as np
from habitat_sim.agent.agent import AgentState, SixDOFPose


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
RANDOM_MOVE="y"
TMP="p"
RESET="r"

def get_shortest_path(sim, samples):
    path_results = []

    for sample in samples:
        path = habitat_sim.ShortestPath()
        path.requested_start = sample[0]
        path.requested_end = sample[1]
        found_path = sim.pathfinder.find_path(path)
        path_results.append((found_path, path.geodesic_distance, path.points))
    print(path_results)
    return path_results

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

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():

    env = habitat.Env(
        config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    )
    
    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    
    count_steps = 0

    agent = ShortestPathFollowerAgent(env=env, goal_radius=1)

    env.current_episode



    while not env.episode_over:
        #habitat_sim.nav.GreedyGeodesicFollower.find_path
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
            action = agent.act(observations)
            print("Agent's position has been reset to a random valid point.")
        elif keystroke == ord(RESET):
            observations = env.reset()
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