from __future__ import division, print_function, absolute_import
import numpy as np

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, PedestrianLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.dynamics import Obstacle, Occlusion

class Pomdp(AbstractEnv):
    """
        A road crossing area with occlusion environment.

        More details to add
    """

    COLLISION_REWARD        = -1
    RIGHT_LANE_REWARD       =  1e-1
    HIGH_VELOCITY_REWARD    =  2e-1
    MERGING_VELOCITY_REWARD = -5e-1
    LANE_CHANGE_REWARD      = -5e-2

    DEFAULT_CONFIG = {"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                      "centering_position": 0.3}

    def __init__(self):
        super(Pomdp, self).__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        self.make_road()
        self.make_vehicles()

    def configure(self, config):
        self.config.update(config)

    def _observation(self):
        return super(Pomdp, self)._observation()

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
                 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity

        return utils.remap(action_reward[action] + reward,
                           [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.vehicle.position[0] > 370

    def reset(self):
        self.make_road()
        self.make_vehicles()
        return self._observation()

    def make_road2(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # pre-processing
        ends = [150, 80, 80, 150]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        W = StraightLane.DEFAULT_WIDTH

        # Highway lanes
        ptr_LB, ptr_UB = 0, ends[0]+ends[1] # Miss what is relevant only for merging lane
        net.add_lane("a", "b", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[c, s])) # Lane 1
        net.add_lane("a", "b", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[n, c]))  # Lane 2
        ptr_LB, ptr_UB = ptr_UB, ptr_UB+ends[1]
        net.add_lane("b", "c", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[c, s]))
        net.add_lane("b", "c", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[n, s]))
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[2]
        net.add_lane("c", "d", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([ptr_LB, W], [ptr_UB, W], line_types=[n, c]))
        
        # Crossing Lane
        # TBA

        # Merging lane
        amplitude = 3.25
        offset = 6.5
        ljk = StraightLane([0, offset + 2*W], [ends[0], offset + 2*W], line_types=[c, c], forbidden=True)
        if 1:
            lkb = SineLane(ljk.position(ends[0], -amplitude),
                           ljk.position(sum(ends[:2]), -amplitude),
                           amplitude, 2 * np.pi / (2*ends[1]),
                           np.pi / 2,
                           line_types=[c, c], forbidden=True)
        else:
            lkb = StraightLane([ends[0], offset + 2*W], [ends[0]+ends[1], 2*W], line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0], line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        
        road = Road(network=net)
        # road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def make_road(self):
        net = RoadNetwork()
    
        # pre-processing
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        W = StraightLane.DEFAULT_WIDTH
    
        # Highway lanes
        ends = [150, 10, 80]
        ptr_LB, ptr_UB = 0, 0
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[0]
        net.add_lane("a", "b", StraightLane([ptr_LB,   W], [ptr_UB,   W], line_types=[c, s]))  # Lane 1
        net.add_lane("a", "b", StraightLane([ptr_LB, 2*W], [ptr_UB, 2*W], line_types=[s, c]))  # Lane 2
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[1]
        net.add_lane("b", "c", StraightLane([ptr_LB,   W], [ptr_UB,   W], line_types=[c, s]))  # Lane 1
        net.add_lane("b", "c", StraightLane([ptr_LB, 2*W], [ptr_UB, 2*W], line_types=[s, c]))  # Lane 2
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[2]
        net.add_lane("c", "d", StraightLane([ptr_LB,   W], [ptr_UB,   W], line_types=[c, s]))  # Lane 1
        net.add_lane("c", "d", StraightLane([ptr_LB, 2*W], [ptr_UB, 2*W], line_types=[s, c]))  # Lane 2
    
        # Crossing Lane
        ends = [2, 2 * W, 2]
        ptr_LB, ptr_UB = 0, 0
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[0]
        lij = PedestrianLane([150, ptr_LB], [150, ptr_UB], width=10)
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[1]
        ljk = PedestrianLane([150, ptr_LB], [150, ptr_UB], width=10)
        ptr_LB, ptr_UB = ptr_UB, ptr_UB + ends[2]
        lkl = PedestrianLane([150, ptr_LB], [150, ptr_UB], width=10)

        net.add_lane("i", "j", lij)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "l", lkl)
    
        road = Road(network=net)
        # road.vehicles.append(Obstacle(road, [150-10, 2*W]))
        road.vehicles.append(Occlusion(road, [150 - 10, 2 * W], [2 * W, W]))
        self.road = road

    def make_vehicles(self):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)

        road.vehicles.append(ego_vehicle)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), velocity=29))
        if 0:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), velocity=29))
            road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), velocity=31))
            road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), velocity=31.5))
    
            merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=20)
            merging_v.target_velocity = 30
            road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
