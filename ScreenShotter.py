import mss
import mss.tools


def take_screenshot():
    im = None
    with mss.mss() as sct:
        top = 235
        lower = 625
        left = 1000
        right = 1600
        bbox = (left, top, right, lower)

        im = sct.grab(bbox)

    return im

