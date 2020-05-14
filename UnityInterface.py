import PyPipeline as pp
import time

class UnityInterface:
    __instance = None

    @staticmethod 
    def getInstance(env_count):
        """ Static access method. """
        if UnityInterface.__instance == None:
            UnityInterface(env_count)

        return UnityInterface.__instance

    def __init__(self, env_count):
        """ Virtually private constructor. """
        if UnityInterface.__instance != None:
            raise Exception("This class is a singleton!")

        self.environment_count = env_count
        self.observations = [None] * env_count
        self.gamestate_updates = 0
        self.sleep_time = 0.1
        UnityInterface.__instance = self
        self.update_observations()
  
    def update_observations(self):
        self.observations = pp.read_observation()
        

    def set_action(self, action_list, env_index):
        pp.write_actions(action_list, env_index)
        time.sleep(self.sleep_time)
        self.update_observations()

    
    def set_gamestate(self, reset_type):
        """isNoteGameOver = 0, isGameOver = 1, isEvalGameOver = 2"""
        self.gamestate_updates += 1
        if self.gamestate_updates == self.environment_count:
            self.gamestate_updates = 0
            pp.write_gameover(reset_type)
            time.sleep(self.sleep_time)
            self.update_observations()
            
    



