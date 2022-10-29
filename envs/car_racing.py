"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from gym.spaces import Box
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

from shapely.geometry import Point, LineString, Polygon
import time
from gym.spaces.discrete import Discrete

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
        self.contact = None
        self.t_last_contact = time.time()

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        self.contact = contact
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
                self.t_last_contact = time.time()
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array", "state_pixels"],
                "video.frames_per_second": FPS}

    def __init__(self, collision_reward=-10, discrete=False, multi_input=True, speed_limit=50, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        if discrete:  # Steer left, steer right, gas, brake
            self.action_space = Discrete(4)
        else:  # steer, gas, brake
            self.action_space = Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32))
            # self.action_space = Box(np.array([-1, 0, 0], dtype=np.float32), np.array([+1, +1, +1], dtype=np.float32))
        if multi_input:
            self.observation_space = spaces.Dict(spaces={"img": Box(0, 255, (STATE_H, STATE_W, 3), dtype=np.uint8),
                                                         "vec": Box(0, 1, shape=(3,), dtype=np.float32)})
        else:
            self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        # Logging rendering variables
        self.collision_reward = collision_reward
        self.action = None
        self.q_action, self.q = -1, [-1, -1, -1, -1]
        self.c_action, self.c = None, [None, None, None, None]
        self.discrete = discrete
        self.state = None
        self.multi_input = multi_input
        # Display functions
        self.score_label, self.q_label, self.col_pred_label, self.action_label = None, None, None, None
        self.chosen_action_label, self.vel_label, self.possible_actions_labels = None, None, None
        self.possible_actions_q_labels, self.possible_actions_c_labels, self.transform = None, None, None

        self.speed_limit = speed_limit

        self.contact_idx = 0  # Closest idx of road tile with respect to car position

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha)
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1: i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(np.square(first_perp_x * (track[0][2] - track[-1][2]))
                                      + np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many instances of this message)")
        self.car = Car(self.world, *self.track[0][1:4])
        self.contact_idx = 0

        return self.step(None)[0]

    def update_critic_values(self, q_action, c_action, q, c):
        self.q_action = q_action
        self.c_action = c_action
        self.q = q
        self.c = c

    def step(self, action):  # Actions: Steer Gas brake
        self.action = [0, 0, 0]
        if action is not None:
            if self.discrete:
                if action == 0:  # Turn Left
                    self.car.steer(-1)
                    self.action[0] = -1
                elif action == 1:  # Turn Right
                    self.car.steer(1)
                    self.action[0] = 1
                elif action == 2:  # Accelerate
                    self.car.gas(1)
                    self.action[1] = 1
                else:  # Brake
                    self.car.brake(0.8)
                    self.action[2] = 1
            else:
                self.car.steer(-action[0])
                self.action[0] = action[0]
                if action[1] <= 0:
                    self.car.brake(-action[1])
                    self.action[2] = -action[1]
                elif action[1] > 0:
                    self.car.gas(action[1])
                    self.action[1] = action[1]

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        velocity = np.linalg.norm(self.car.hull.linearVelocity)

        if self.multi_input:
            car = self.car.hull
            vec = np.array([velocity, car.angularVelocity, self.car.wheels[0].angle - car.angle], dtype=np.float32)
            self.state = {"img": self.render("state_pixels"), "vec": vec}
        else:
            self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            if self.speed_limit is not None:
                if velocity < self.speed_limit:
                    self.reward += 0.005 * velocity
                else:
                    self.reward -= 0.01 * velocity
            else:
                self.reward += 0.005 * velocity  # Max velocity = 100 km/h

            step_reward = (self.reward - self.prev_reward) * 0.1
            self.prev_reward = self.reward

            if self.tile_visited_count == len(self.track):
                done = True

            # Idx of track is linearly increasing. Thus, check idx in +-2 range of last closest contact idx
            car_pos = Point(self.car.hull.position)
            track_distance = np.inf
            low, high = self.contact_idx - 2, self.contact_idx + 3
            for i in range(low, high):
                i = i % len(self.road)
                polygon_road = Polygon(self.road[i].fixtures[0].shape.vertices)
                distance = polygon_road.distance(car_pos)
                if distance < track_distance:
                    track_distance = distance
                    self.contact_idx = i

            # Reset environment, when lane is left
            if not done and track_distance > 0.5:
                done = True
                step_reward = self.collision_reward

            # Reset environment, when duration between now and last new tile visit is too long ago
            # if time.time() - self.contactListener_keepref.t_last_contact > 10:
            if self.t * FPS > 350 and velocity < 0.1:
                done = True
                step_reward = self.collision_reward
                print("Car timed out. Did not visit new tile frequently enough")

            # Find distance from crash by computing distance to short side of the tile which points outward the track:
            distance_before_crash = 0
            if track_distance < 0.5:
                tile = self.road[self.contact_idx].fixtures[0].shape
                edge_0 = np.array(tile.vertices[0])
                min_distance = np.inf
                min_idx = 0

                # Dirty fix of error, when tile has not 4 corner
                distance_before_crash = 5
                if len(tile.vertices) == 4:
                    # Finding closest neighbour vertex for 0
                    for i in range(1, 4):
                        edge_i = np.array(tile.vertices[i])
                        distance = np.linalg.norm(edge_0 - edge_i)
                        if distance < min_distance:
                            min_distance = distance
                            min_idx = i

                    # Assigning lines
                    line_1, line_2 = [], []
                    for i in range(4):
                        if i == 0 or i == min_idx:
                            line_1.append(Point(tile.vertices[i]))
                        else:
                            line_2.append(Point(tile.vertices[i]))
                    line_1 = LineString(line_1)
                    line_2 = LineString(line_2)

                    distance_before_crash = min(line_1.distance(car_pos), line_2.distance(car_pos)) + 0.5

            info = {'track_progress': self.tile_visited_count / len(self.track),
                    'velocity': velocity,
                    'distance_before_crash': distance_before_crash,
                    'collision_pred': -1 if self.c_action is None else self.c_action,
                    }

        return self.state, step_reward, done, info

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            debug_kwargs = {"font_size": 17, "anchor_x": "left", "anchor_y": "center", "color": (255, 255, 255, 255),
                            "bold": True}

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label("0000", font_size=36, x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x="left",
                                                 anchor_y="center", color=(255, 255, 255, 255), )
            self.q_label = pyglet.text.Label("Q: 1", x=WINDOW_W - 250, y=WINDOW_H - 30, **debug_kwargs)
            self.col_pred_label = pyglet.text.Label("Col: 2", x=WINDOW_W - 250, y=WINDOW_H - 70, **debug_kwargs)
            self.action_label = pyglet.text.Label("Steer, gas, brake", x=WINDOW_W - 250, y=WINDOW_H - 110,
                                                  **debug_kwargs)
            self.chosen_action_label = pyglet.text.Label("[0.1  , 2.1  ,  3.1]", x=WINDOW_W - 250, y=WINDOW_H - 140,
                                                         **debug_kwargs)
            self.vel_label = pyglet.text.Label("Vel: 2", x=WINDOW_W - 250, y=WINDOW_H - 180, **debug_kwargs)

            # Make Text smaller
            debug_kwargs["font_size"] = 13
            self.possible_actions_labels = pyglet.text.Label("Left, Right, Gas Brake", x=WINDOW_W - 250,
                                                             y=WINDOW_H - 220, **debug_kwargs)
            self.possible_actions_q_labels = pyglet.text.Label("0.0, 0.0, 0.0, 0.0", x=WINDOW_W - 250, y=WINDOW_H - 250,
                                                               **debug_kwargs)
            self.possible_actions_c_labels = pyglet.text.Label("0.0, 0.0, 0.0, 0.0", x=WINDOW_W - 250, y=WINDOW_H - 280,
                                                               **debug_kwargs)
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)), )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (win.context._nscontext.view().backingScaleFactor())  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        # if mode != "state_pixels":
        render_value_function = False if mode == "state_pixels" else True
        self.render_indicators(WINDOW_W, WINDOW_H, render_value_function)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (pyglet.image.get_buffer_manager().get_color_buffer().get_image_data())
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [+PLAYFIELD, +PLAYFIELD, 0, +PLAYFIELD, -PLAYFIELD, 0, -PLAYFIELD, -PLAYFIELD, 0, -PLAYFIELD,
                     +PLAYFIELD, 0, ]
        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend([k * x + k, k * y + 0, 0,  # polygon: x1, y1, z1, x2,y2, z2, ....
                                  k * x + 0, k * y + 0, 0,
                                  k * x + 0, k * y + k,
                                  0, k * x + k, k * y + k, 0])
        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors))  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H, render_value_function=True):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        if render_value_function:
            # Add box around logging functions like q, col prob, action
            polygons.extend([WINDOW_W - 270, WINDOW_H, 0,
                             WINDOW_W - 270, WINDOW_H - 300, 0,
                             WINDOW_W, WINDOW_H - 300, 0,
                             WINDOW_W, WINDOW_H, 0])
            colors.extend([0, 0, 0, 1] * 4)

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend([place * s, h + h * val, 0,
                             (place + 1) * s, h + h * val, 0,
                             (place + 1) * s, h, 0,
                             (place + 0) * s, h, 0, ])

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend([(place + 0) * s, 4 * h, 0,
                             (place + val) * s, 4 * h, 0,
                             (place + val) * s, 2 * h, 0,
                             (place + 0) * s, 2 * h, 0, ])

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(len(polygons) // 3, ("v3f", polygons), ("c4f", colors))  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

        if render_value_function:
            self.q_label.text = "Q: {:.2f}".format(self.q_action)
            self.q_label.draw()
            self.action_label.text = "Steer, gas, brake"
            self.action_label.draw()

            if self.action[0] < 0:
                self.chosen_action_label.text = \
                    "[L {:.1f}  , {:.1f}  ,  {:.1f}]".format(abs(self.action[0]), self.action[1], self.action[2])
            else:
                self.chosen_action_label.text = \
                    "[R {:.1f}  , {:.1f}  ,  {:.1f}]".format(self.action[0], self.action[1], self.action[2])

            self.chosen_action_label.draw()
            self.vel_label.text = "Vel: {:.2f}".format(true_speed)
            self.vel_label.draw()

            if self.c_action is not None:
                self.col_pred_label.text = "Crash: {:.4f}".format(self.c_action)
                self.col_pred_label.draw()

            # Evaluate all actions
            self.possible_actions_labels.text = "    Left, Right, Gas, Brake"
            self.possible_actions_labels.draw()

            # Sometimes self.c = List[np.array((1,1))]
            if isinstance(self.c[0], np.ndarray):
                c = (self.c[0].item(), self.c[1].item(), self.c[2].item(), self.c[3].item())
                q = (self.q[0].item(), self.q[1].item(), self.q[2].item(), self.q[3].item())
            else:
                c = (self.c[0], self.c[1], self.c[2], self.c[3])
                q = (self.q[0], self.q[1], self.q[2], self.q[3])

            self.possible_actions_q_labels.text = "Q: [ {:.1f} , {:.1f} , {:.1f} , {:.1f} ]".format(*q)
            self.possible_actions_q_labels.draw()
            if self.c_action is not None:
                self.possible_actions_c_labels.text = "C: [ {:.1f} , {:.1f} , {:.1f} , {:.1f} ]".format(*c)
                self.possible_actions_c_labels.draw()


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            # a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            a[1] = -0.8


    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            # a[2] = 0
            a[1] = 0


    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
