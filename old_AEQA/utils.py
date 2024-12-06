#class object to json



def env_to_json(env):
    image_uuid = str(uuid.uuid4())
    image_name = image_uuid + ".png"
    agent_state = env.sim.get_agent_state()
    position = agent_state.position.tolist()  # Convert to list for JSON serialization
    rotation = agent_state.rotation  # Quaternion
    rotation_euler = rotation.to_euler()  # Convert quaternion to Euler angles

    location_data = {
        "image_uuid": image_uuid,
        "agent_state": {
            "position": position,
            "rotation": {
                "x": rotation_euler.x,
                "y": rotation_euler.y,
                "z": rotation_euler.z,
            }
        }
    }
    return json_data


