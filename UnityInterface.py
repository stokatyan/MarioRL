import PyPipeline as pp
import time

class UnityInterface:
    __instance = None

    @staticmethod 
    def getInstance(env_count):
        """ Static access method. """
        if UnityInterface.__instance == None:
            UnityInterface(env_count)

        UnityInterface.__instance.init_count += 1
        return UnityInterface.__instance

    def __init__(self, env_count):
        """ Virtually private constructor. """
        if UnityInterface.__instance != None:
            raise Exception("This class is a singleton!")

        self.init_count = -2
        self.environment_count = env_count
        self.gamestate = 0
        self.actions = [None] * env_count
        self.observations = [None] * env_count
        self.action_updates = 0
        self.sleep_time = 0.1
        UnityInterface.__instance = self
        self.update_observations()
  
    def update_observations(self):
        self.observations = pp.read_observation()

    def write_actions(self):
        pp.write_actions(self.actions)

    def set_action(self, action_list, env_index):
        self.actions[env_index] = action_list
        self.action_updates += 1

        if self.action_updates == self.environment_count:
            self.action_updates = 0
            self.write_actions()
            time.sleep(self.sleep_time)
            self.update_observations()
            
    



