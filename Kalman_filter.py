import pygame
import math
import numpy as np
import random
from maze import *
from Simulation import *

class Kalman_filter:
    def __init__(self, process_noise, measurement_noise):
        self.state = np.array([[ball_x], [ball_y], [ball_angle]])
        self.covariance =  np.eye(3) * 1.0 #start initializing covariance with identity matrix, can be changed to sigma**2
        self.R = process_noise #can describe this too
        self.Q = measurement_noise #can describe this too according to the slides

    # Kinematics
    v = (right_wheel_speed + left_wheel_speed) / 2
    omega = (right_wheel_speed - left_wheel_speed) / wheel_base
    u = np.array([[v], omega])
    z = np.array([[cell.x], [cell.y], [ball_angle]])  # need to retrieve correct cell coordinates
    ball_angle += omega
    dx = v * math.cos(ball_angle)
    dy = v * math.sin(ball_angle)
    dt = int(max(abs(dx), abs(dy)) / 0.1) + 1 #copied this from your code
    ball_angle_change = omega * dt

    def predict(self, u):
        A = np.eye(3) * 1.0
        B = np.array([[dt * np.cos(ball_angle), 0], [dt * np.sin(ball_angle), 0], [0, dt]])
        self.state = self.state * A + B * u
        self.covariance += self.R

    def correct(self, z):
        C = np.eye(3) * 1.0
        K = self.covariance / (self.covariance + self.R) #Kalman gain
        self.state = self.state + K * (z - C * self.state)
        self.covariance = (1 - K * C) * self.covariance

        return self.state, self.covariance


#GPT
""""
def kalman_predict(state, covariance, v, omega, dt):
    dx = v * math.cos(state[2, 0]) * dt
    dy = v * math.sin(state[2, 0]) * dt
    dtheta = omega * dt

    state[0, 0] += dx
    state[1, 0] += dy
    state[2, 0] += dtheta

    motion_noise = 0.01
    covariance += np.eye(3) * motion_noise
    return state, covariance


def kalman_correct(state, covariance, measured_distance, landmark_x, landmark_y):
    expected_dx = landmark_x - state[0, 0] #would need landmark_x to be x-coordinate of cell
    expected_dy = landmark_y - state[1, 0]
    expected_distance = math.sqrt(expected_dx ** 2 + expected_dy ** 2)

    innovation = measured_distance - expected_distance

    # Simplified gain
    K = 0.5
    state[0, 0] += K * innovation * (expected_dx / expected_distance)
    state[1, 0] += K * innovation * (expected_dy / expected_distance)

    # Reduce uncertainty
    covariance *= (1 - K)

    return state, covariance
"""