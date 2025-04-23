import pygame
import math
import numpy as np
import random
from maze import *
from Simulation import *

"""
# Kinematics
v = (right_wheel_speed + left_wheel_speed) / 2
omega = (right_wheel_speed - left_wheel_speed) / wheel_base
u = np.array([[v], [omega]])
z = np.array([[cell.x], [cell.y], [ball_angle]])  # need to retrieve correct cell coordinates
ball_angle += omega
dx = v * math.cos(ball_angle)
dy = v * math.sin(ball_angle)
dt = int(max(abs(dx), abs(dy)) / 0.1) + 1 #copied this from your code
ball_angle_change = omega * dt
"""""

class Kalman_filter:
    def __init__(self):
        self.state = np.array([[ball_x], [ball_y], [ball_angle]])
        self.covariance =  np.eye(3) * 1.0 #start initializing covariance with identity matrix, can be changed to sigma**2
        self.R = np.eye(3) * 0.5 #process_noise
        self.Q = np.eye(3) * 0.5 #measurement_noise

    def predict(self, u):
        #need to define u as the action chosen
        A = np.eye(3) * 1.0
        B = np.array([[dt * np.cos(ball_angle), 0], [dt * np.sin(ball_angle), 0], [0, dt]])
        self.state = self.state * A + B * u
        self.covariance += self.R
        #draw predicted path/localization

    def correct(self, z):
        #need to define z by using the landmark measurements
        C = np.eye(3) * 1.0
        K = self.covariance / (self.covariance + self.R) #Kalman gain, dont remember how i got to this
        self.state = self.state + K * (z - C * self.state)
        self.covariance = (1 - K * C) * self.covariance
        #draw the corrected path

        return self.state, self.covariance

#how to do it for several landmarks at once (i.e., triangulation)