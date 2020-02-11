from gym.spaces import Dict
import sawyer_control.envs.sawyer_pushing as sawyer_pushing
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv
import numpy as np
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SawyerPushXYEnv(sawyer_pushing.SawyerPushXYEnv, MultitaskEnv):
    ''' Must Wrap with Image Env to use!'''

    def __init__(self,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        sawyer_pushing.SawyerPushXYEnv.__init__(self, **kwargs)
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = None
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def compute_rewards(self, actions, obs):
        raise NotImplementedError('Use Image based reward')

    def _get_obs(self):
        achieved_goal = None
        state_obs = super()._get_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def reset(self):
        if self.action_mode == "position":
            self.in_reset = True
            self._position_act(self.pos_control_reset_position - self._get_endeffector_pose())
            self.in_reset = False
        else:
            self._reset_robot()
        if self.pause_on_reset:
            if self.use_gazebo_auto:
                print(bcolors.OKBLUE+'move object to reset position and press enter'+bcolors.ENDC)
                if self.random_init:
                    obj_pos_rand = np.random.uniform(
                        self.goal_space.low,
                        self.goal_space.high,
                        size=(1, self.goal_space.low.size),
                    )
                    obj_pos = obj_pos_rand[0][:2]
                    self.pos_object_reset_position[:2] = obj_pos
                args = dict(x=self.pos_object_reset_position[0],
                            y=self.pos_object_reset_position[1],
                            z=self.pos_object_reset_position[2])
                msg = dict(func='set_object_los', args=args)
                self.client.sending(msg, sleep_before=self.config.SLEEP_BEFORE_SENDING_CMD_SOCKET,
                                    sleep_after=self.config.SLEEP_BETWEEN_2_CMDS)
                self.client.sending(msg, sleep_before=0,
                                    sleep_after=self.config.SLEEP_AFTER_SENDING_CMD_SOCKET)
            else:
                input(bcolors.OKBLUE+'move object to reset position and press enter'+bcolors.ENDC)
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goal(self):
        return MultitaskEnv.sample_goal(self)

    def sample_goals(self, batch_size):
        goals = super().sample_goals(batch_size)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        super().set_to_goal(goal)


if __name__ == "__main__":
    env = SawyerPushXYEnv()
    env.reset()
    env.set_to_goal({'state_desired_goal': [0, .1]})
