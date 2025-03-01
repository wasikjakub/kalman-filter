import pandas as pd
import numpy as np
from kalman_filter import KalmanFilter
from utils import plot_all_dimensions

train_dataset = False
# Load data
data_path = 'data/test.csv'
data = pd.read_csv(data_path)

# Sampling time
dt = data["Time"][1] - data["Time"][0]

# Initialize the EKF with gyroscope data
# Initialize the EKF with accelerometer as the primary predictor
kf = KalmanFilter(dt)

# Run the Kalman filter on each time step
predicted_roll = []
predicted_pitch = []
predicted_yaw = []

for i in range(len(data)):
    z_acc = np.array([data['AccX'][i], data['AccY'][i], data['AccZ'][i]])  # Accelerometer measurements
    z_gyro = np.array([data['GyroX'][i], data['GyroY'][i], data['GyroZ'][i]])  # Gyroscope measurements
    roll_pred, pitch_pred, yaw_pred = kf.update(z_acc, z_gyro)

    predicted_roll.append(roll_pred)
    predicted_pitch.append(pitch_pred)
    predicted_yaw.append(yaw_pred)

# Convert to arrays if needed
predicted_roll = np.array(predicted_roll)
predicted_pitch = np.array(predicted_pitch)
predicted_yaw = np.array(predicted_yaw)


if train_dataset:
    plot_all_dimensions(data["pitch"], data["roll"], data["yaw"], data["Time"], predicted_pitch, predicted_roll, predicted_yaw)
    mse = 0
    for i in range(len(predicted_pitch)):
        mse += (data['pitch'][i] - predicted_pitch[i]) ** 2
        mse += (data['roll'][i] - predicted_roll[i]) ** 2
        mse += (data['yaw'][i] - predicted_yaw[i]) ** 2
    mse /= 3 * (len(predicted_pitch))
    print(f"Minimum square qrror: {mse}")

else:
    plot_all_dimensions(predicted_pitch, predicted_roll, predicted_yaw, data["Time"])

    ### Save to file
    data = {
        "Id": np.arange(1, len(predicted_roll) + 1),
        "pitch": predicted_pitch,
        "roll": predicted_roll,
        "yaw": predicted_yaw
    }
    df = pd.DataFrame(data)
    df.to_csv("predicted_data.csv", index=False)