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
        self.covariance = np.eye(3) * 5  # initial uncertainty
        self.R = np.eye(3) * 5  # process noise
        self.Q = np.eye(2) * 1  # measurement noise (2D: distance, bearing)
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
                robot_xy=(self.state[0,0], self.state[1,0]),
                landmark_xy=lm,
                phi_meas=f0["measurement"][1]
            )
            self.state[2, 0] = theta_est

            # 4. Kovarianz ggf. zurücksetzen oder vergrößern,
            #    da du gerade komplett neu initialisierst:
            self.covariance = np.eye(3) * 5  

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



    def get_observed_features(self, ball_x, ball_y, ball_angle,
                              max_sensor_length=100, fov=np.pi):
        observed_features = []
        x, y, theta = ball_x, ball_y, ball_angle

        sigma_r = math.sqrt(self.Q[0, 0])
        sigma_phi = math.sqrt(self.Q[1, 1])

        for cell in self.grid:
            if not getattr(cell, "marker", False):
                continue

            # Landmark-Position
            lx, ly = cell.x, cell.y
            dx, dy = lx - x, ly - y
            dist_true = math.hypot(dx, dy)
            if dist_true > max_sensor_length:
                continue

            # *** KORREKT : richtiger Messwert mit Rauschen ***
            # true bearing relativ zur Robot-Richtung
            phi_true = (math.atan2(dy, dx) - theta + math.pi) % (2*math.pi) - math.pi

            measured_distance = dist_true + np.random.normal(0, sigma_r)
            measured_bearing  = phi_true   + np.random.normal(0, sigma_phi)

            # Visualisierung
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

    def intersect_two_circles(self, p1, r1, p2, r2, eps=1e-9):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy)
        # keine Lösung bei zu großer/kleiner Entfernung oder identischem Zentrum
        if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
            return None
        # Abstand von p1 zur Mittellinie
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        # Höhe des Dreiecks, robust clamped
        h2 = r1*r1 - a*a
        if h2 < -eps:
            return None
        h = math.sqrt(max(h2, 0.0))
        # Mittelpunkts-Koordinaten
        xm = x1 + a * dx / d
        ym = y1 + a * dy / d
        # Verschiebung entlang der Normalen
        rx = -dy * (h / d)
        ry = dx * (h / d)
        # Tangentialfall: ein Schnittpunkt
        if h < eps:
            return [(xm, ym)]
        # zwei Schnittpunkte
        return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

    def angle_diff(self, a, b):
        # normalisiere auf [-pi, pi]
        return (a - b + math.pi) % (2*math.pi) - math.pi

    def estimate_theta_from_landmark(self, robot_xy, landmark_xy, phi_meas):
        dx = landmark_xy[0] - robot_xy[0]
        dy = landmark_xy[1] - robot_xy[1]
        theta = math.atan2(dy, dx) - phi_meas
        return (theta + math.pi) % (2*math.pi) - math.pi

    def triangulate_position_from_two_landmarks(
        self, lm1, r1, phi1, lm2, r2, phi2,
        prev_xy=None, epsilon=1e-6
    ):
        # 1. beide Schnittpunkte berechnen
        ints = self.intersect_two_circles(lm1, r1, lm2, r2)
        if not ints:
            return None

        # 2. optional: Schnittpunkte nach Nähe zu letzter Pose sortieren
        if prev_xy is not None:
            ints = sorted(ints,
                          key=lambda p: (p[0]-prev_xy[0])**2 + (p[1]-prev_xy[1])**2)

        best = None
        best_err = float('inf')

        # 3. Standard-Loop über Kandidaten (der erste ist dann der nächste am prev_xy)
        for (x, y) in ints:
            # a) globale Blickwinkel
            pred1 = math.atan2(lm1[1] - y, lm1[0] - x)
            pred2 = math.atan2(lm2[1] - y, lm2[0] - x)
            # b) Schätzung von θ als Kreis-Mittelwert von pred−φ
            d1 = self.angle_diff(pred1, phi1)
            d2 = self.angle_diff(pred2, phi2)
            theta_i = math.atan2(math.sin(d1)+math.sin(d2),
                                 math.cos(d1)+math.cos(d2))
            # c) Winkel-Residuen
            err1 = abs(self.angle_diff(pred1 - theta_i, phi1))
            err2 = abs(self.angle_diff(pred2 - theta_i, phi2))
            err = err1 + err2

            if err < best_err - epsilon:
                best_err = err
                best = (x, y, theta_i)

        if best is None:
            return None

        x_sel, y_sel, theta_sel = best
        theta_sel = (theta_sel + math.pi) % (2*math.pi) - math.pi
        print(f"  Using lm1={lm1} with φ₁={phi1:.3f}, lm2={lm2} with φ₂={phi2:.3f}")

        return np.array([x_sel, y_sel, theta_sel])


    def triangulate_position_from_landmarks(self, observed, max_distance=100):
        valid = []
        for f in observed:
            lm_id = f["id"]
            if lm_id in self.landmark_map:
                r, phi = f["measurement"]
                if r <= max_distance:
                    valid.append((self.landmark_map[lm_id], r, phi))

        # Zu wenige Landmarken → kein Ergebnis
        if len(valid) < 2:
            return None

        # Genau zwei Landmarken: nimm Schnittpunkt, der näher an letzter Pose liegt
        if len(valid) == 2:
            (lm1, r1, phi1), (lm2, r2, phi2) = valid
            prev_xy = (self.state[0, 0], self.state[1, 0])
            return self.triangulate_position_from_two_landmarks(
                lm1, r1, phi1,
                lm2, r2, phi2,
                prev_xy=prev_xy
            )

        # Mehr als zwei Landmarken: Least-Squares-Lösung
        (x0, y0), r0, _ = valid[0]
        A, b = [], []
        for (xi, yi), ri, _ in valid[1:]:
            A.append([2*(xi - x0), 2*(yi - y0)])
            b.append([r0*r0 - ri*ri - x0*x0 + xi*xi - y0*y0 + yi*yi])
        A = np.array(A)
        b = np.array(b).reshape(-1, 1)
        try:
            (x, y), *_ = np.linalg.lstsq(A, b, rcond=None)
            return np.array([x.item(), y.item()])
        except np.linalg.LinAlgError:
            return None


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