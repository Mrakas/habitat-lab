import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs

def set_random_initial_state(simulator, agent):
    # Step 1: 获取随机的可导航点
    random_position = simulator.sample_navigable_point()
    
    # Step 2: 生成随机旋转（单位四元数）
    random_rotation = quat_from_coeffs(np.random.randn(4))
    random_rotation /= np.linalg.norm(random_rotation)  # 确保单位化

    # Step 3: 创建 AgentState
    random_state = habitat_sim.agent.AgentState(
        position=random_position,
        rotation=random_rotation,
    )

    # Step 4: 设置 Agent 的状态
    agent.set_state(random_state)
    print(f"Agent initialized to position: {random_position}, rotation: {random_rotation}")

# 使用示例
def example():
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = "data/scene_datasets/habitat_example/apartment_1.glb"  # 替换为你的场景路径
    
    # 创建一个默认的 AgentConfiguration
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = []  # 定义传感器（如 RGB, 深度等）
    
    # 创建模拟器
    cfg = habitat_sim.Configuration(sim_config, [agent_config])
    simulator = habitat_sim.Simulator(cfg)

    # 获取代理
    agent = simulator.get_agent(0)

    # 设置随机初始状态
    set_random_initial_state(simulator, agent)

    # 测试观察
    observations = simulator.get_sensor_observations()
    print("Observations acquired.")
    
    simulator.close()

if __name__ == "__main__":
    example()
