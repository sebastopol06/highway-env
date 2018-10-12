from __future__ import division, print_function
import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.logger import Loggable


class Vehicle(Loggable):
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_VELOCITIES = [20, 21]
    """ Range for random initial velocities [m/s] """

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1, np_random=None, prepend=False):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :param np.random.RandomState np_random: a random number generator
        :return: A vehicle with random position and/or velocity
        """
        if np_random is None:
            np_random = np.random
        default_spacing = 30
        _from = np_random.choice(list(road.network.graph.keys()))
        _to = np_random.choice(list(road.network.graph[_from].keys()))
        _id = np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        if prepend:
            x0 = np.min([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 0
            x0 -= offset
        else:
            x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
            x0 += offset
        velocity = velocity or np_random.randint(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        v = cls(road, road.network.get_lane((_from, _to, _id)).position(x0, 0), 0, velocity)
        return v

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity)
        return v

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -self.velocity

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)

    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True

    def to_dict(self, origin_vehicle=None):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading)
        }
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position):
        super(Obstacle, self).__init__(road, position, velocity=0)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH

class Occlusion(Vehicle):
    """
        A motionless complex obstacle at a given position, leading to occlusion
    """

    def __init__(self, road, position, size):#, points):
        super(Occlusion, self).__init__(road, position, velocity=0) # , points, velocity=0)
        self.target_velocity = 0
        self.LENGTH, self.WIDTH = size
