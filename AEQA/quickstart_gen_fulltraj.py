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
from utils import DestinationDataLoader

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
Y="y"
TMP="p"
TEST="t"

def get_shortest_path(sim, samples): # get key points only.
    path_results = []

    for sample in samples:
        path = habitat_sim.ShortestPath()
        path.requested_start = sample[0]
        path.requested_end = sample[1]
        found_path = sim.pathfinder.find_path(path)
        path_results.append((found_path, path.geodesic_distance, path.points))
    print("result:",path_results)
    return path_results

def from_json_to_state(json_path: str) -> AgentState:
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    position = np.array(data['start_position'])
    rotation = quaternion.quaternion(*data['start_rotation'])
    
    sensor_states = {}
    
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
    #env.current_episode
    
    data_loader = DestinationDataLoader('/home/marcus/workplace/habitat-lab/AEQA/data/destinations')

    print("agent_state-------------", env.sim.get_agent_state())

    for file_path in data_loader:

            
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
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        if data['shortest_paths'] is not None: # 如果已经计算过最短路径，跳过
            print("pass",file_path)
            continue
        print("file_path:",file_path)

        #修改起点
        env.sim.set_agent_state(position=data['start_position'], rotation=quaternion.quaternion(*data['start_rotation']))
        print("ENV-rotation====",env.sim.get_agent_state().rotation)
        print("my-rotation====",data['start_rotation'])
        print("data_loader",data_loader)
        #修改终点
        env.current_episode.goals[0].position = data['goals'][0]['position']

        shortest_paths =[]
        #迭代修改坐标
        #append路径

        while not env.episode_over:
            #habitat_sim.nav.GreedyGeodesicFollower.find_path
            keystroke = cv2.waitKey(0)
            shortest_paths.append(env.sim.get_agent_state().position.tolist())
            shortest_paths.append(quaternion.as_float_array(env.sim.get_agent_state().rotation).tolist())
            #import ipdb; ipdb.set_trace()
            print(env.sim.get_agent_state().position.tolist())
            #print("shortest_paths",shortest_paths)

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
            elif keystroke == ord(Y):
                action = agent.act(observations)
                print("towards gt")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.stop
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue
            observations = env.step(action)
            count_steps += 1

            # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            #     observations["pointgoal_with_gps_compass"][0],
            #     observations["pointgoal_with_gps_compass"][1]))
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        import ipdb; ipdb.set_trace()
        data['shortest_paths'] = shortest_paths
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print("Episode finished after {} steps.".format(count_steps))
        print("json name:",file_path)


if __name__ == "__main__":
    example()