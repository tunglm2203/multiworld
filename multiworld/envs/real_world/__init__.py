import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_real_worl_envs():
    global REGISTERED
    if REGISTERED:
        LOGGER.info("[WARNING] sawyer_control environment already registered")
        return
    REGISTERED = True
    LOGGER.info("Registering real_world gym environments")
    from multiworld.envs.mujoco.cameras import (
        sawyer_init_camera_zoomed_in
    )

    """
    Reaching tasks
    """

    register(
        id='SawyerReachXYZReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_reaching:SawyerReachXYZEnv',
        kwargs={
            'fix_goal': False,
            'reward_type': 'hand_distance',
            'action_mode': 'position',
            'max_speed': 0.5,
        },
    )

    register(
        id='SawyerPushXYReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_pushing:SawyerPushXYEnv',
        kwargs={
            'fix_goal': False,
            'action_mode': 'position',
            'max_speed': 0.5,
        },
    )

    register(
        id='SawyerDoorReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_door:SawyerDoorEnv',
        kwargs={
            'fix_goal': False,
            'action_mode': 'position',
            'max_speed': 0.5,
        },
    )
