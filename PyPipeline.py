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
    return read_json(observation_file)


def write_action(values):
    """
    Converts a given array of values to an action json that the agent can respond to.
    Contents of the array should correspend to: [left, right, up down].

    The final json is written to the action file
    """
    kLeft = 'left'
    kRight = 'right'
    kUp = 'up'
    kDown = 'down'

    actions = {}
    if len(values) is 4:
        actions[kLeft] = values[0] is 1
        actions[kRight] = values[1] is 1
        actions[kUp] = values[2] is 1
        actions[kDown] = values[3] is 1

    json_str = json.dumps(actions)
    
    file = open(f'{path}{action_file}', 'w')
    file.write(f'{json_str}')
    
    file.close()    


def write_reset_game():
    file = open(f'{path}{action_file}', 'w')
    file.write('{"action":2}')
    
    file.close()


def read_screenshot(should_flatten=False):
    sct_img = ss.take_screenshot()
    im = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    im = im.convert("L")
    im = im.resize((20, 200))
    data = asarray(im).T.reshape(20, 200, 1)
    data = data/255
    if should_flatten:
        data = data.flatten()

    return data


def debug_screenshot(size=None):
    sct_img = ss.take_screenshot()
    im = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    im = im.convert("L")
    if size is not None:
        im = im.resize(size)
    im.show()


def debug_shape():
    data = read_screenshot()
    print(data.shape)
