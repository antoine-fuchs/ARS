import pygame
import math
import numpy as np
import random

from tornado.queues import Queue

from maze import *
#from Simulation import *

class KalmanFilter:
    def __init__(self):
        from Simulation import ball_x, ball_y,ball_angle
        self.state = np.array([[ball_x], [ball_y], [ball_angle]])
        self.covariance =  np.eye(3) * 0.5 #start initializing covariance with sigma**2 --> should try out different values (local vs global initialization)
        self.R = np.eye(3) * 0.5 #process_noise
        self.Q = np.eye(3) * 0.5 #measurement_noise
        self.grid = grid #dont know if i need this but maybe for the trail?
        self.estimated_position = []
        self.max_estimated_position_length = 500
        #do we need a step counter?


    def predict(self, dt = 0.1):
        # Kinematics --> repurposed from simulation
        v = (right_wheel_speed + left_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base #rate of rotation
        ball_angle += omega #ball angle == theta
        dx = v * math.cos(ball_angle) * dt
        dy = v * math.sin(ball_angle) * dt
        ball_angle_change = omega * dt

        # Initializing A, B & u
        u = np.array([[v], [omega]]) #action chosen
        A = np.eye(3) * 1.0
        B = np.array([[dt * np.cos(ball_angle), 0], [dt * np.sin(ball_angle), 0], [0, dt]])

        # Prediction of new state (after action u) & covariance
        self.state = self.state * A + B * u #or should I save this in a new predicted_state vector?
        self.covariance += self.R #same question here...

        # Drawing the estimated position trail
        self.estimated_position.append(self,state[:2].copy()) #why copy?
        estimated_position_trail = pygame.draw.circle(screen, BLUE, pos, 2)
        if len(self.estimated_position) > self.max_estimated_position_length:
            del(self.estimated_position[0])
        return estimated_position_trail


    def get_observed_features(self):
        observed_features = []
        #input should be: z = Cell(x, y, cell_id)
        for z in self.grid: #should actually also be able to use features not part of the grid
            dx = z[0] - self.state[0]
            dy = z[1] - self.state[1]

            true_distance = np.sqrt(dx ** 2 + dy ** 2) #dont fully understand why we need this or what it does
            if true_distance < max_sensor_length: #daniel has it the other way around but this doesnt make sense to me
                continue

            # Compute actual angle & bearing vs measured data (with noise Q added)
            angle = np.arctan2(dy, dx)
            bearing = angle - self.state[2] #phi, according to lecture slides
            measured_distance = true_distance + self.Q
            measured_bearing = bearing + self.Q

            observed_data = {
                "position": np.array(z),
                "measurement": np.array([measured_distance, measured_bearing])}
            observed_features.append(observed_data)
            return observed_features
            # Draw line from robot to landmark
            collision, _ = check_wall_collision(probe_x, probe_y, ball_radius, grid)
            if collision:
                pygame.draw.line(screen, ORANGE, (sensor_start_x, sensor_start_y), (z[0], z[1]), 2)


    def correct(self, observed_features):
        x, y, ball_angle = self.state #predicted state estimation

        for feature in observed_features:
            fx, fy = feature["position"]
            measured_distance, measured_bearing = feature["measurement"]

        # Expected feature measurements based on predicted state
        dx = fx - x
        dy = fy - y
        q = dx ** 2 + dy ** 2
        expected_distance = np.sqrt(q)
        expected_bearing = np.arctan2(dy, dx) - ball_angle
        predicted_z = np.array([expected_distance, expected_bearing])

        z = np.array([measured_distance, measured_bearing]) #or should i be using coordinates instead: #z = np.array([[cell.x], [cell.y], [ball_angle]])  --> need to retrieve correct cell coordinates
        C = np.eye(3) * 1.0
        K = self.covariance * (self.covariance + self.Q) #Kalman gain

        # Calculating the corrected state & covariance
        self.state = self.state + K * (z - C * self.state)
        self.covariance = (1 - K * C) * self.covariance
        return self.state, self.covariance
        # draw the corrected path
