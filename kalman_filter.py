import numpy as np
import math


class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(3)  # State [roll, pitch, yaw]
        self.P = np.eye(3) * 1e-1000  # Initial estimation error covariance
        self.Q = np.eye(3) * 1e-4  # Process noise covariance
        self.R = np.eye(3) * 5e-1  # Measurement noise covariance

    def adjustment(self):
        newx0 = self.x[0] - self.x[1]/4000
        newx1 = self.x[1] + self.x[0]/4000 - self.x[2]/8000
        self.x[1] = newx1
        self.x[0] = newx0

    def convert_acc(self, acc):
        roll = math.atan2(acc[1], acc[2]) * (180 / math.pi) + 2.8640772537203656
        pitch = math.atan2(-acc[0], (math.sqrt((acc[1] ** 2) + (acc[2] ** 2)))) * (180 / math.pi)
        # yaw = math.atan2(acc[1], acc[0]) * (180 / math.pi) if acc[0] != 0 else 0
        yaw = self.x[2] + 0.08
        return np.array([roll, pitch, yaw])

    def f(self, x, gyro):
        roll_rate, pitch_rate, yaw_rate = gyro[:3] /1000
        roll_pred = x[0] + roll_rate * self.dt
        pitch_pred = x[1] + pitch_rate * self.dt
        yaw_pred = x[2] + yaw_rate * self.dt

        x_pred = np.array([roll_pred, pitch_pred, yaw_pred])
        F = np.eye(3)  # Transition matrix
        return x_pred, F

    def h(self, x):
        h_x = x
        H = np.eye(3)  # Measurement matrix
        return h_x, H

    def predict(self, gyro):
        """
        Predicts the next state using gyroscope data.
        """
        x_pred, F = self.f(self.x, gyro)
        self.P = F @ self.P @ F.T + self.Q
        self.x = x_pred

    def update(self, acc, gyro):
        """
        Performs the update step using accelerometer data to refine the state.
        """
        self.predict(gyro)

        # Use accelerometer data as measurement
        acc_measurement = self.convert_acc(acc)

        # Compute residuals and Kalman gain
        h, H = self.h(self.x)
        y = acc_measurement - h  # Residual

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

        self.adjustment()

        return self.x[0], self.x[1], self.x[2]  # Roll, Pitch, Yaw