import pygame
import math
import numpy as np
from Maze import *  # ensure this provides needed classes like `check_wall_collision`

estimated_position = []

def gaussian_prob(error, sigma_sq):
    return (1.0 / math.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * (error ** 2) / sigma_sq)


class KalmanFilter:
    def __init__(self, initial_state, grid, screen, estimated_state = [[None],[None],[None]]):
        self.state = np.array(initial_state).reshape((3, 1))  # [[x], [y], [theta]]
        self.covariance = np.eye(3) * 0.5  # initial uncertainty
        self.R = np.eye(3) * 3  # process noise
        self.Q = np.eye(2) * 3  # measurement noise (2D: distance, bearing)
        #self.estimated_state 

        #self.estimated_position = []
        self.max_estimated_position_length = 500

        self.grid = grid  # landmarks or features
        self.screen = screen  # pygame screen for drawing

        self.landmark_map = {}  # maps cell_id to (x, y)
        for cell in grid:
            if hasattr(cell, "marker") and cell.marker:
                self.landmark_map[cell.cell_id] = (cell.x, cell.y)

    def predict(self,ball_x,ball_y,ball_angle, dt=0.1):
        from Simulation import right_wheel_speed, left_wheel_speed, wheel_base, RED
        """
        Instead of dead‐reckoning, use only landmark observations to
        predict (i.e. re‐initialize) the robot's position via triangulation.
        """
        # 1. Sammle aktuelle Landmarken‐Messungen
        observed = self.get_observed_features(ball_x,ball_y,ball_angle)
        print(observed)
        
        # 2. Versuche, Position (x,y) per Triangulation zu schätzen
        pos_est = self.triangulate_position_from_landmarks(observed)
        if pos_est is not None:
            # Update state x,y direkt aus Landmarken‐Triangulation
            self.state[0, 0], self.state[1, 0] = pos_est[0], pos_est[1]
            
            # 3. Falls gewünscht, schätze theta aus einer einzelnen Landmarke
            #    (optional – hier am Beispiel der ersten gesehenen Landmarke)
            f0 = observed[0]
            lm = self.landmark_map[f0["id"]]
            theta_est = self.estimate_theta_from_landmark(
                robot_pos=(self.state[0,0], self.state[1,0]),
                landmark_pos=lm,
                measured_bearing=f0["measurement"][1]
            )
            self.state[2, 0] = theta_est

            # 4. Kovarianz ggf. zurücksetzen oder vergrößern,
            #    da du gerade komplett neu initialisierst:
            self.covariance = np.eye(3) * 500  

        else:
            # Wenn zu wenige Landmarken, behalte letzte Schätzung
            # und erhöhe Unsicherheit
            self.covariance += self.R

        # 5. Trail‐Handling wie gehabt
        estimated_pos = (int(self.state[0, 0]), int(self.state[1, 0]))
        estimated_position.append(estimated_pos)
        if len(estimated_position) > self.max_estimated_position_length:
            estimated_position.pop(0)
        for pos in estimated_position:
            pygame.draw.circle(self.screen, RED, pos, 2)
        print(estimated_pos)


    def get_observed_features(self,ball_x,ball_y,ball_angle, max_sensor_length=100, fov=np.pi):
        """
        Returns a list of observed features in the form:
        {
            'id': marker_id,
            'measurement': np.array([distance, bearing]),
            'true_position': (x, y)  ← optional, for debugging only
        }
        """
        observed_features = []
        x, y, theta = ball_x,ball_y,ball_angle

        for cell in self.grid:
            if not getattr(cell, "marker", False):
                continue

            # Use cell center as landmark position
            landmark_x = cell.x
            landmark_y = cell.y

            dx = landmark_x - x
            dy = landmark_y - y
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance > max_sensor_length:
                continue

            # Simulated noisy measurement
            measured_distance = distance + np.random.normal(0, self.Q[0, 0])
            measured_bearing = ball_angle + np.random.normal(0, self.Q[1, 1])

            # Visualization (optional)
            pygame.draw.line(self.screen, (0, 255, 0), (int(x), int(y)), (int(landmark_x), int(landmark_y)), 1)
            pygame.draw.circle(self.screen, (255, 165, 0), (int(landmark_x), int(landmark_y)), 4)

            # Append observation
            observed_features.append({
                "id": cell.cell_id,
                "measurement": np.array([measured_distance, measured_bearing]),
                "true_position": (landmark_x, landmark_y)  # ← for debugging/visualization only
            })

        return observed_features


    def compute_landmark_likelihood(self, feature, landmark_pos, robot_pose, landmark_id, sigma_r2, sigma_phi2, sigma_s2=1):
        x, y, theta = robot_pose
        r_meas, phi_meas = feature["measurement"]
        s_meas = feature["id"]

        dx = landmark_pos[0] - x
        dy = landmark_pos[1] - y

        r_hat = np.sqrt(dx**2 + dy**2)
        phi_hat = math.atan2(dy, dx) - theta
        phi_hat = (phi_hat + np.pi) % (2 * np.pi) - np.pi

        p_r = gaussian_prob(r_meas - r_hat, sigma_r2)
        p_phi = gaussian_prob(phi_meas - phi_hat, sigma_phi2)
        p_s = gaussian_prob(s_meas - landmark_id, sigma_s2)

        return p_r * p_phi * p_s


    def correct(self, observed_features):
        sigma_r2 = self.Q[0, 0]
        sigma_phi2 = self.Q[1, 1]

        for feature in observed_features:
            marker_id = feature["id"]
            z = feature["measurement"]

            if marker_id not in self.landmark_map:
                continue

            fx, fy = self.landmark_map[marker_id]
            x, y, theta = self.state.flatten()

            # Berechne Likelihood q
            q = self.compute_landmark_likelihood(
                feature=feature,
                landmark_pos=(fx, fy),
                robot_pose=(x, y, theta),
                landmark_id=marker_id,
                sigma_r2=sigma_r2,
                sigma_phi2=sigma_phi2
            )

            if q < 1e-4:
                continue  # zu unwahrscheinlich → ignorieren

            dx = fx - x
            dy = fy - y
            q_val = dx ** 2 + dy ** 2
            expected_distance = np.sqrt(q_val)
            expected_bearing = math.atan2(dy, dx) - theta
            expected_bearing = (expected_bearing + np.pi) % (2 * np.pi) - np.pi

            z_pred = np.array([[expected_distance], [expected_bearing]])
            z = z.reshape((2, 1))

            C = np.zeros((2, 3))
            C[0, 0] = -dx / expected_distance
            C[0, 1] = -dy / expected_distance
            C[1, 0] = dy / q_val
            C[1, 1] = -dx / q_val
            C[1, 2] = -1

            S = C @ self.covariance @ C.T + self.Q
            K = self.covariance @ C.T @ np.linalg.inv(S)

            self.state = self.state + K @ (z - z_pred)
            self.covariance = (np.eye(3) - K @ C) @ self.covariance

        return self.state, self.covariance

    def triangulate_position_from_landmarks(self, observed_features, max_distance=100):
        """
        Estimate position (x, y) using triangulation from ≥2 landmark observations within range.
        If exactly 2 are available, also return theta via bearing.
        """
        # Filter only those within max_distance
        valid = []
        for f in observed_features:
            landmark_id = f["id"]
            if landmark_id not in self.landmark_map:
                continue
            r = f["measurement"][0]
            if r <= max_distance:
                valid.append(f)

        if len(valid) < 2:
            return None  # not enough

        # If exactly 2 → triangulate and estimate theta
        if len(valid) == 2:
            f1, f2 = valid
            id1, id2 = f1["id"], f2["id"]
            x1, y1 = self.landmark_map[id1]
            x2, y2 = self.landmark_map[id2]
            r1, r2 = f1["measurement"][0], f2["measurement"][0]

            # Linearize the equations
            A = np.array([[2 * (x2 - x1), 2 * (y2 - y1)]])
            b = np.array([[r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2]])

            try:
                pos_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                x, y = pos_est.flatten()

                # Schätze theta aus erster Landmarke
                lm = self.landmark_map[id1]
                measured_phi = f1["measurement"][1]
                theta = self.estimate_theta_from_landmark((x, y), lm, measured_phi)

                return np.array([x, y, theta])  # Full pose
            except:
                return None

        # If ≥3 valid landmarks → normal least squares triangulation
        A = []
        b = []
        for i in range(len(valid) - 1):
            f1 = valid[i]
            f2 = valid[i + 1]
            id1, id2 = f1["id"], f2["id"]
            x1, y1 = self.landmark_map[id1]
            x2, y2 = self.landmark_map[id2]
            r1, r2 = f1["measurement"][0], f2["measurement"][0]

            A.append([2 * (x2 - x1), 2 * (y2 - y1)])
            b.append([r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2])

        A = np.array(A).reshape(-1, 2)
        b = np.array(b).reshape(-1, 1)

        try:
            pos_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return pos_est.flatten()  # Nur x, y
        except:
            return None

    
    def estimate_theta_from_landmark(self, robot_pos, landmark_pos, measured_bearing):
        """
        robot_pos: np.array([x, y])
        landmark_pos: np.array([mx, my])
        measured_phi: gemessener Winkel (relativ)
        """
        dx = landmark_pos[0] - robot_pos[0]
        dy = landmark_pos[1] - robot_pos[1]
        global_angle = math.atan2(dy, dx)

        theta_estimated = global_angle - measured_bearing

        # Normalize to [-π, π]
        theta_estimated = (theta_estimated + np.pi) % (2 * np.pi) - np.pi

        return theta_estimated


    def initialize_pose_from_landmarks(self, observed_features):
        """
        Estimate full pose (x, y, θ) using triangulation and one bearing.
        """
        pos = self.triangulate_position_from_landmarks(observed_features)
        if pos is None:
            return False  # Initialization failed

        # Estimate θ from first valid landmark
        for f in observed_features:
            if f["id"] in self.landmark_map:
                landmark = self.landmark_map[f["id"]]
                measured_bearing = f["measurement"][1]
                theta = self.estimate_theta_from_landmark(pos, landmark, measured_bearing)
                self.state = np.array([[pos[0]], [pos[1]], [theta]])
                return True  # Success

        return False