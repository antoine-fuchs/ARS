import pygame
import math
import numpy as np

from maze import *  # ensure this provides needed classes like `check_wall_collision`
estimated_position = []

class KalmanFilter:
    def __init__(self, initial_state, grid, screen):
        self.state = np.array(initial_state).reshape((3, 1))  # [[x], [y], [theta]]
        self.covariance = np.eye(3) * 0.5  # initial uncertainty
        self.R = np.eye(3) * 10  # process noise
        self.Q = np.eye(2) * 10  # measurement noise (2D: distance, bearing)

        #self.estimated_position = []
        self.max_estimated_position_length = 500

        self.grid = grid  # landmarks or features
        self.screen = screen  # pygame screen for drawing

    def predict(self, dt=0.1):
        from Simulation import right_wheel_speed, left_wheel_speed, wheel_base, RED


        # Compute linear and angular velocity
        v = (right_wheel_speed + left_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base

        # Update heading (theta)
        theta = self.state[2, 0]
        theta += omega * dt

        # Define control vector and matrices
        u = np.array([[v], [omega]])
        A = np.eye(3)
        B = np.array([
            [dt * math.cos(theta), 0],
            [dt * math.sin(theta), 0],
            [0, dt]
        ])

        # Predict the new state and covariance
        self.state = A @ self.state + B @ u
        self.covariance = A @ self.covariance @ A.T + self.R

        # Store the estimated position as a tuple of integers
        estimated_pos = (int(self.state[0, 0]), int(self.state[1, 0]))
        estimated_position.append(estimated_pos)

        # Limit trail length if necessary
        if len(estimated_position) > self.max_estimated_position_length:
            estimated_position.pop(0)

        # Draw all points in the estimated position trail
        for pos in estimated_position:
            pygame.draw.circle(self.screen, RED, pos, 2)




    def get_observed_features(self, max_sensor_length = 100, fov=4*np.pi):
        """
        Simulates and visualizes observations of nearby real landmarks (cells with marker=True)
        based on the current estimated state.
        """
        observed_features = []

        x, y, theta = self.state.flatten()

        for z in self.grid:
            if not hasattr(z, "marker") or not z.marker:
                continue

            dx = z.x - x
            dy = z.y - y

            print(x,y)
            

            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance > max_sensor_length:
                continue


            angle_to_feature = np.arctan2(dy, dx)
            relative_bearing = angle_to_feature - theta
            relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi

            if abs(relative_bearing) > fov / 2:
                continue

            # Simulate noisy measurements
            measured_distance = distance + np.random.normal(0, self.Q[0, 0])
            measured_bearing = relative_bearing + np.random.normal(0, self.Q[1, 1])

            # Draw line to detected feature (green)
            pygame.draw.line(
                self.screen,
                (0, 255, 0),  # green
                (int(x), int(y)),
                (int(z.x), int(z.y)),
                1
            )

            # Draw circle on detected feature (orange)
            pygame.draw.circle(
                self.screen,
                (255, 165, 0),  # orange
                (int(z.x), int(z.y)),
                4
            )

            observed_features.append({
                "position": np.array([z.x, z.y]),
                "measurement": np.array([measured_distance, measured_bearing])
            })

        return observed_features


    def correct(self, observed_features):
        for feature in observed_features:
            fx, fy = feature["position"]
            measured_distance, measured_bearing = feature["measurement"]

            x, y, theta = self.state.flatten()
            dx = fx - x
            dy = fy - y
            q = dx ** 2 + dy ** 2
            expected_distance = np.sqrt(q)
            expected_bearing = np.arctan2(dy, dx) - theta

            z_pred = np.array([[expected_distance], [expected_bearing]])
            z = np.array([[measured_distance], [measured_bearing]])

            # Observation matrix: maps state to expected measurement (linearized)
            C = np.zeros((2, 3))
            C[0, 0] = -dx / expected_distance
            C[0, 1] = -dy / expected_distance
            C[1, 0] = dy / q
            C[1, 1] = -dx / q
            C[1, 2] = -1

            # Kalman gain
            S = C @ self.covariance @ C.T + self.Q
            K = self.covariance @ C.T @ np.linalg.inv(S)

            # Update state and covariance
            self.state = self.state + K @ (z - z_pred)
            self.covariance = (np.eye(3) - K @ C) @ self.covariance

        return self.state, self.covariance
