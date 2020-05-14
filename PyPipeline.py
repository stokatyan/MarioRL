import json
import ScreenShotter as ss
from PIL import Image
import numpy as np
from numpy import asarray
import time

path = 'MarioRLScene/Assets/Resources/'

observation_file = 'environment_output.txt'
gamestate_file = 'environment_state.txt'
action_file = 'environment_input.txt'

def read_json(filename):
    obs = {}
    with open(f'{path}{filename}') as mytxt:
        for line in mytxt:
            if is_json(line):
                obs = json.loads(line)
    return obs


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True


def read_gamestate():
    return read_json(gamestate_file)


def read_observation():
    items_json = read_json(observation_file)
    observation_array = items_json['Items']
    return observation_array


def write_actions(values):
    """
    Converts a given list of array of values to an action json that the agent can respond to.
    Contents of the given array should correspend to: [[left, right, up down]].

    The final json is written to the action file
    """
    kLeft = 'left'
    kRight = 'right'
    kUp = 'up'
    kDown = 'down'

    directions = [kLeft, kRight, kUp, kDown]

    actions = []
    for actionList in values:
        act = {}
        for index in range(4):
            val = actionList[index]
            act[directions[index]] = float(val)
        
        actions.append(act)

    json_str = json.dumps(actions)

    json_str = "{\"Items\":" + json_str
    json_str = json_str + "}"
    
    file = open(f'{path}{action_file}', 'w')
    file.write(f'{json_str}')
    
    file.close()


def write_no_action():
    write_action([0, 0, 0, 0])


def write_gameover(reset_type):
    dictionary = {}
    dictionary['gameover'] = reset_type
    json_str = json.dumps(dictionary)
    
    file = open(f'{path}{gamestate_file}', 'w')
    file.write(f'{json_str}')
    
    file.close()
    

def read_screenshot(should_flatten=False):
    sct_img = ss.take_screenshot()
    im = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    # im = im.convert("L")
    im = im.resize((100, 65))
    data = asarray(im).T.reshape(100, 65, 3)
    data = data/255
    if should_flatten:
        data = data.flatten()

    return data


def debug_screenshot(size=None):
    sct_img = ss.take_screenshot()
    im = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    # im = im.convert("L")
    if size is not None:
        im = im.resize(size)
    im.show()


def debug_shape():
    data = read_screenshot()
    print(data.shape)
