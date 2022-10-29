"""
2D rendering framework
"""
import os
import sys

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL installed. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )

import math
import numpy as np
from gym.envs.classic_control.rendering import Viewer

RAD2DEG = 57.29577951308232


class CustomViewer(Viewer):
    def __init__(self, width, height, display=None):
        super(CustomViewer, self).__init__(width, height, display)
        debug_kwargs = {"font_size": 14, "anchor_x": "left", "anchor_y": "center", "color": (0, 0, 0, 255),
                        "bold": False}

        self.q_action, self.c_action = -1, -1
        self.q, self.c = [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]
        self.reward = -1
        self.velocity = -1
        self.crash_distance = -1

        self.q_action_label = pyglet.text.Label("Q: 1", x=0, y=30, **debug_kwargs)
        self.c_action_label = pyglet.text.Label("C: 1", x=0, y=10, **debug_kwargs)

        self.action_label = pyglet.text.Label("Down, Right , Up  , Left", x=150, y=50, **debug_kwargs)
        self.q_label = pyglet.text.Label("Q: 1", x=110, y=30, **debug_kwargs)
        self.c_label = pyglet.text.Label("C: 1", x=110, y=10, **debug_kwargs)

        self.reward_label = pyglet.text.Label("Reward: 200", x=230, y=80, **debug_kwargs)
        self.crash_distance_label = pyglet.text.Label("C dist: 9", x=450, y=10, **debug_kwargs)
        self.velocity_label = pyglet.text.Label("V: 2", x=450, y=30, **debug_kwargs)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()

        self.render_text()

        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []

        return arr if return_rgb_array else self.isopen

    def render_text(self) -> None:
        '''
        Taken from https://stackoverflow.com/questions/44604391/pyglet-draw-text-into-texture
        @return:
        '''
        glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # glColor3f(1.0, 1.0, 1.0)

        # Statistics of taken action
        self.q_action_label.text = "Q: {:.0f}".format(self.q_action)
        self.c_action_label.text = "C: {:.2f}".format(self.c_action)
        self.q_action_label.draw()
        self.c_action_label.draw()

        # Sometimes self.c = List[np.array((1,1))]
        if isinstance(self.c[0], np.ndarray):
            c = (self.c[0].item(), self.c[1].item(), self.c[2].item(), self.c[3].item())
            q = (self.q[0].item(), self.q[1].item(), self.q[2].item(), self.q[3].item())
        else:
            c = (self.c[0], self.c[1], self.c[2], self.c[3])
            q = (self.q[0], self.q[1], self.q[2], self.q[3])
        self.c_label.text = "C: [ {:.1f} , {:.1f} , {:.1f} , {:.1f} ]".format(*c)
        self.q_label.text = "Q: [ {:.0f} , {:.0f} , {:.0f} , {:.0f} ]".format(*q)
        self.c_label.draw()
        self.q_label.draw()
        self.action_label.draw()
        # System state
        self.reward_label.text = "Reward: {:.0f}".format(self.reward)
        self.crash_distance_label.text = "Crash dist {:.1f}".format(self.crash_distance)
        self.velocity_label.text = "Vel: {:.1f}".format(self.velocity)
        self.reward_label.draw()
        self.crash_distance_label.draw()
        self.velocity_label.draw()

    def update_critic_values(self, q_action, c_action, q, c):
        self.q, self.c, self.q_action, self.c_action = q, c, q_action, c_action

    def update_state_values(self, reward, velocity, crash_distance):
        self.reward, self.velocity, self.crash_distance = reward, velocity, crash_distance
