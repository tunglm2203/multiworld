from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_gridworld_envs():
    global REGISTERED
    if REGISTERED:
        return

    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")

    register(
        id='GoalGridworld-v0',
        entry_point='multiworld.envs.gridworlds.goal_gridworld:GoalGridworld',
    )
    register(
        id='GoalGridworld-Concatenated-v0',
        entry_point='multiworld.envs.gridworlds.goal_gridworld:GoalGridworld',
        kwargs={'concatenated': True}
    )
