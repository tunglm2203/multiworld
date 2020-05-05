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
    max_speed = 0.1
    register(
        id='SawyerReachXYZReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_reaching:SawyerReachXYZEnv',
        kwargs={
            'fix_goal': True,
            'reward_type': 'hand_distance',
            'action_mode': 'position',
            'max_speed': max_speed,
            'config_name': 'tung_config',
        },
    )

    register(
        id='SawyerPushXYReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_pushing:SawyerPushXYEnv',
        kwargs={
            'fix_goal': False,  # True
            # 'fixed_goal': (.50, -0.1, .45, .0),
            'action_mode': 'position',
            'max_speed': max_speed,
            'config_name': 'tung_config',
            'hand_goal_low': (0.55, -0.1),
            'hand_goal_high': (0.65, 0.1),
            'puck_goal_low': (0.5, -0.15),
            'puck_goal_high': (0.7, 0.15)
        },
    )

    register(
        id='SawyerPushXYRealMedium-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_pushing:SawyerPushXYEnv',
        kwargs={
            'fix_goal': False,  # True
            # 'fixed_goal': (.50, -0.1, .45, .0),
            'action_mode': 'position',
            'max_speed': max_speed,
            'config_name': 'tung_config',
            'hand_goal_low': (0.55, -0.05),
            'hand_goal_high': (0.65, 0.05),
            'puck_goal_low': (0.5, -0.2),
            'puck_goal_high': (0.7, 0.2)
        },
    )

    register(
        id='SawyerPushXYRealHarder-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_pushing:SawyerPushXYEnv',
        kwargs={
            'fix_goal': False,  # True
            # 'fixed_goal': (.50, -0.1, .45, .0),
            'action_mode': 'position',
            'max_speed': max_speed,
            'config_name': 'tung_config',
            'hand_goal_low': (0.5, -0.2),
            'hand_goal_high': (0.7, 0.2),
            'puck_goal_low': (0.5, -0.2),
            'puck_goal_high': (0.7, 0.2)
        },
    )

    register(
        id='SawyerDoorReal-v0',
        entry_point='multiworld.envs.real_world.sawyer.sawyer_door:SawyerDoorEnv',
        kwargs={
            'fix_goal': True,
            'action_mode': 'position',
            'max_speed': max_speed,
        },
    )
