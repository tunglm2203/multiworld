from multiworld.envs.mujoco import register_mujoco_envs
from multiworld.envs.pygame import register_pygame_envs
from multiworld.envs.gridworlds import register_gridworld_envs
from multiworld.envs.real_world import register_real_worl_envs


def register_all_envs():
    register_mujoco_envs()
    register_pygame_envs()
    register_gridworld_envs()
    register_real_worl_envs()
