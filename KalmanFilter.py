import pygame
import math
import numpy as np
from Maze import * 

estimated_position = []

def gaussian_prob(error, sigma_sq):
    return (1.0 / math.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * (error ** 2) / sigma_sq)


class KalmanFilter:
    def __init__(self, initial_state, grid, screen, R=None, Q=None, trail_length=500):
        # state: [x, y, theta]^T
        self.state = np.array(initial_state).reshape((3,1))
        self.covariance = np.eye(3) * 0.01
        # process noise
        self.R = R if R is not None else np.eye(3) * 0.01
        # measurement noise (range, bearing)
        self.Q = Q if Q is not None else np.diag([1.0, 1.0])


        self.grid = grid

        # for visualizing estimated trajectory
        self.estimated_trail = []
        self.max_trail_length = trail_length
        self.screen = screen

        # landmarks map from id to (x,y)
        self.landmark_map = {cell.cell_id: (cell.x, cell.y)
                             for cell in grid if getattr(cell, 'marker', False)}
        

    def predict(self, v, omega, dt):

        theta = self.state[2,0]
        B = np.array([[dt * math.cos(theta), 0],
                    [dt * math.sin(theta), 0],
                    [0,                   dt]])
        u = np.array([[v], [omega]])
        self.state = self.state + B @ u


        self.covariance = self.covariance + self.R


        x_f, y_f = self.state[0,0], self.state[1,0]
        self.estimated_trail.append((x_f, y_f))
        if len(self.estimated_trail) > self.max_trail_length:
            self.estimated_trail.pop(0)

        for x_f, y_f in self.estimated_trail:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),
                (x_f, y_f),
                1
            )





    def get_observed_features(self, ball_x, ball_y, ball_angle,
                              max_sensor_length=100, fov=np.pi):
        observed_features = []
        x, y, theta = ball_x, ball_y, ball_angle

        sigma_r = math.sqrt(self.Q[0, 0])
        sigma_phi = math.sqrt(self.Q[1, 1])

        for cell in self.grid:
            if not getattr(cell, "marker", False):
                continue

            # Landmark position
            lx, ly = cell.x, cell.y
            dx, dy = lx - x, ly - y
            dist_true = math.hypot(dx, dy)
            if dist_true > max_sensor_length:
                continue

            # true bearing relative to robot's orientation
            phi_true = (math.atan2(dy, dx) - theta + math.pi) % (2*math.pi) - math.pi

            measured_distance = dist_true + np.random.normal(0, sigma_r)
            measured_bearing  = phi_true   + np.random.normal(0, sigma_phi)

            # Visualization of sensor ray and landmark
            pygame.draw.line(self.screen, (0,255,0),
                             (int(x),int(y)), (int(lx),int(ly)), 1)
            pygame.draw.circle(self.screen, (255,165,0),
                               (int(lx),int(ly)), 4)

            observed_features.append({
                "id": cell.cell_id,
                "measurement": np.array([measured_distance, measured_bearing]),
                "true_position": (lx, ly)
            })
        return observed_features

    def correct(self, z, landmark_pos):
        """
        Correction step:
          K_t = Σ̄_t C^T (C Σ̄_t C^T + Q_t)^{-1}
          μ_t = μ̄_t + K_t (z_t - C μ̄_t)
          Σ_t = (I - K_t C) Σ̄_t
        Here z = [range; bearing], and C is the linearized measurement Jacobian.
        """
        x, y, theta = self.state.flatten()
        lx, ly = landmark_pos

        # Expected measurement h(μ̄)
        dx = lx - x; dy = ly - y
        r_pred = math.hypot(dx, dy)
        phi_pred = math.atan2(dy, dx) - theta
        phi_pred = (phi_pred + math.pi) % (2*math.pi) - math.pi
        h = np.array([[r_pred], [phi_pred]])

        # Jacobian C (2×3)
        C = np.zeros((2,3))
        C[0,0] = -dx / r_pred
        C[0,1] = -dy / r_pred
        C[1,0] =  dy / (r_pred**2)
        C[1,1] = -dx / (r_pred**2)
        C[1,2] = -1

        # Innovation y = z - h
        z = z.reshape((2,1))
        y_innov = z - h
        y_innov[1,0] = (y_innov[1,0] + math.pi) % (2*math.pi) - math.pi

        # Compute Kalman gain
        S = C @ self.covariance @ C.T + self.Q
        K = self.covariance @ C.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ y_innov
        self.state[2,0] = (self.state[2,0] + math.pi) % (2*math.pi) - math.pi
        self.covariance = (np.eye(3) - K @ C) @ self.covariance


    def intersect_two_circles(self, p1, r1, p2, r2, eps=1e-9):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy)
        # keine oder unendliche Lösungen?
        if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
            return None

        # Abstand vom ersten Zentrum zur Verbindungsgeraden
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        h2 = r1*r1 - a*a
        if h2 < -eps:
            return None
        h = math.sqrt(max(h2, 0.0))

        # Mittlerer Punkt auf der Geraden zwischen den Zentren
        xm = x1 + a * dx / d
        ym = y1 + a * dy / d

        # Offsets für die beiden Schnittpunkte
        rx = -dy * (h / d)
        ry =  dx * (h / d)

        if h < eps:
            # Genau ein Schnittpunkt (Tangent)
            return [(xm, ym)]
        # Zwei Schnittpunkte
        return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

    def angle_diff(self, a, b):
        return (a - b + math.pi) % (2*math.pi) - math.pi

    def estimate_theta_from_landmark(self, robot_xy, landmark_xy, phi_meas):
        dx = landmark_xy[0] - robot_xy[0]
        dy = landmark_xy[1] - robot_xy[1]
        theta = math.atan2(dy, dx) - phi_meas
        return (theta + math.pi) % (2*math.pi) - math.pi

    def triangulate_position_from_two_landmarks(
        self,
        lm1, r1, phi1,
        lm2, r2, phi2,
        prev_xy=None,
        angle_weight=1.0,
        distance_weight=0.0
    ):


        ints = self.intersect_two_circles(lm1, r1, lm2, r2)
        if not ints:
            return None

        best, best_err = None, float('inf')
        px, py = (None, None) if prev_xy is None else prev_xy

        for (x, y) in ints:

            theta1 = self.estimate_theta_from_landmark((x, y), lm1, phi1)
            theta2 = self.estimate_theta_from_landmark((x, y), lm2, phi2)

            theta = math.atan2(
                math.sin(theta1) + math.sin(theta2),
                math.cos(theta1) + math.cos(theta2)
            )
            theta = (theta + math.pi) % (2*math.pi) - math.pi

            pred_phi1 = (math.atan2(lm1[1] - y, lm1[0] - x) - theta + math.pi) % (2*math.pi) - math.pi
            pred_phi2 = (math.atan2(lm2[1] - y, lm2[0] - x) - theta + math.pi) % (2*math.pi) - math.pi

            err_angle = abs(self.angle_diff(pred_phi1, phi1)) + abs(self.angle_diff(pred_phi2, phi2))

            if prev_xy is not None:
                err_dist = (x - px)**2 + (y - py)**2
            else:
                err_dist = 0.0

            err_total = angle_weight * err_angle + distance_weight * err_dist

            if err_total < best_err:
                best_err = err_total
                best = (x, y, theta)

        return np.array(best) if best is not None else None

    def triangulate_position_from_landmarks(self, observed):
        valid = []
        for f in observed:
            lm_id = f['id']
            if lm_id in f['landmark_map']:
                r, phi = f['measurement']
                valid.append((f['landmark_map'][lm_id], r, phi))
        if len(valid) < 2:
            return None
        if len(valid) == 2:
            (lm1, r1, phi1), (lm2, r2, phi2) = valid
            prev_xy = (self.state[0,0], self.state[1,0])
            return self.triangulate_position_from_two_landmarks(lm1, r1, phi1, lm2, r2, phi2, prev_xy)
        x0, y0 = valid[0][0]
        r0 = valid[0][1]
        A, b = [], []
        for (xi, yi), ri, _ in valid[1:]:
            A.append([2*(xi - x0), 2*(yi - y0)])
            b.append([r0*r0 - ri*ri - x0*x0 + xi*xi - y0*y0 + yi*yi])
        A = np.array(A)
        b = np.array(b).reshape(-1,1)
        try:
            (x, y), *_ = np.linalg.lstsq(A, b, rcond=None)
            return np.array([x.item(), y.item(), 0.0])
        except np.linalg.LinAlgError:
            return None

    def initialize_from_landmarks(self, observed, landmark_map):
        """
        Initialize pose by triangulating position from at least two landmarks.
        observed: list of {'id', 'measurement': np.array([r, phi])}
        landmark_map: {id: (x, y)}
        """
        # compute position and optional theta
        pose = self.triangulate_position_from_landmarks(observed)
        if pose is None:
            return False
        # if pose includes theta, use it; else estimate from first landmark
        if len(pose) == 3:
            x0, y0, theta0 = pose
        else:
            x0, y0 = pose
            # estimate orientation from first observed
            lm_id = observed[0]['id']
            r, phi = observed[0]['measurement']
            theta0 = self.estimate_theta_from_landmark((x0, y0), landmark_map[lm_id], phi)
        self.state = np.array([[x0], [y0], [theta0]])
        return True