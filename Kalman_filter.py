import pygame
import math
import numpy as np
import random
from maze import *
from Simulation import *


class Kalman_filter:
    def __init__(self):
        self.state = np.array([[ball_x], [ball_y], [ball_angle]])
        self.covariance =  np.eye(3) * 1.0 #start initializing covariance with identity matrix, can be changed to sigma**2
        self.R = np.eye(3) * 0.5 #process_noise
        self.Q = np.eye(3) * 0.5 #measurement_noise

    def predict(self, dt):
        # Kinematics
        v = (right_wheel_speed + left_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base #rate of rotation
        ball_angle += omega #ball angle == theta
        #dx = v * math.cos(ball_angle) * dt
        #dy = v * math.sin(ball_angle) * dt
        ball_angle_change = omega * dt

        # Initializing A, B & u
        u = np.array([[v], [omega]]) #action chosen
        A = np.eye(3) * 1.0
        B = np.array([[dt * np.cos(ball_angle), 0], [dt * np.sin(ball_angle), 0], [0, dt]])

        # Prediction of new state after action u & covariance
        self.state = self.state * A + B * u
        self.covariance += self.R
        #draw predicted path/localization

    def get_observed_features(self):
        observed_features = []
        #z = Cell(x, y, cell_id)
        for z in grid: #should actually also be able to use features not part of the grid
            dx = z[0] - self.state[0]
            dy = z[1] - self.state[1]

            true_range = np.sqrt(dx ** 2 + dy ** 2)
            if true_range > 2 * ball_radius:  #why >?
                continue
            angle = np.arctan2(dy, dx)
            bearing = angle - self.state[2]
            observed_data = {
                "position": np.array(z),
                "measurement": np.array([true_range, bearing])}

            #if intersection with sensor...
            observed_features.append(observed_data)
            return observed_features
            #draw line from robot to landmark by changing the sensor's color

    def correct(self, observed_features):
        x, y, ball_angle = self.state #predicted state estimation

        for feature in observed_features:
            fx, fy = feature["position"]

            measured_range, measured_bearing = feature["measurement"]

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
