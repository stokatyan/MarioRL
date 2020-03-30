import mss
import mss.tools


def take_screenshot():
    im = None
    with mss.mss() as sct:
        top = 225
        lower = 620
        left = 1285
        right = 1335
        bbox = (left, top, right, lower)

        im = sct.grab(bbox)

    return im

