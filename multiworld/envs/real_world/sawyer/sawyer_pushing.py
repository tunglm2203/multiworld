from gym.spaces import Dict
import sawyer_control.envs.sawyer_pushing as sawyer_pushing
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv
import time
import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
## http://gazebosim.org/tutorials/?tut=ros_comm
## http://answers.gazebosim.org/question/22125/how-to-set-a-models-position-using-gazeboset_model_state-service-in-python/


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

    AUTO_MOVE_OBJ = True

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


    def set_obj_to_pos(self,name, pos):
        #rospy.init_node('set_pose')
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = float(pos[0])
        state_msg.pose.position.y = float(pos[1])
        state_msg.pose.position.z = float(pos[2])
        state_msg.pose.orientation.x = 0.0 #pos[3]
        state_msg.pose.orientation.y = 1.0 #pos[4]
        state_msg.pose.orientation.z = 0.0 #pos[5]
        state_msg.pose.orientation.w = 0.0 #pos[6]
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

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
            if self.AUTO_MOVE_OBJ:
                print(bcolors.OKGREEN+'Auto move object to reset position'+bcolors.ENDC)
                pos = list([self.pos_object_reset_position[0],self.pos_object_reset_position[1],self.pos_object_reset_position[2]])
                self.set_obj_to_pos('cylinder',pos)
                
            else:
                input(bcolors.OKBLUE+'Move object to reset position and press enter'+bcolors.ENDC)
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
        obj_goal = np.concatenate((goal[:2], [self.z]))
        ee_goal = np.concatenate((goal[2:4], [self.z]))
        if self.AUTO_MOVE_OBJ:
            print(bcolors.OKGREEN + 'Auto place object at end effector location' +
                  bcolors.ENDC)
            # TUNG: +- 0.05 to avoid collision since object right below gripper
            x=goal[0] + 0.05
            y=goal[1] - 0.05
            z=self.pos_object_reset_position[2]
            self.set_obj_to_pos('cylinder',[x,y,z])

        else:
            input(bcolors.OKBLUE + 'Place object at end effector location and press enter' +
                  bcolors.ENDC)
        self._position_act(ee_goal - self._get_endeffector_pose()[:3])


if __name__ == "__main__":
    env = SawyerPushXYEnv()
    env.reset()
    env.set_to_goal({'state_desired_goal': [0, .1]})
