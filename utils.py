import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_all_dimensions(pitch, roll, yaw, time, pitch_pred=None, roll_pred=None, yaw_pred=None):
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, pitch, label="Pitch real", color='g')
    if pitch_pred is not None:
        plt.plot(time, pitch_pred, label="Pitch predicted",color='orange')
    plt.xlabel("Time")
    plt.ylabel("Pitch (degrees)")
    plt.title("Pitch")
    plt.legend()

    # Pitch subplot
    plt.subplot(3, 1, 2)
    plt.plot(time, roll, label="Roll real", color='g')
    if roll_pred is not None:
        plt.plot(time, roll_pred, label="Roll predicted",color='orange')
    plt.xlabel("Time")
    plt.ylabel("Roll (degrees)")
    plt.title("Roll")
    plt.legend()

    # Yaw subplot
    plt.subplot(3, 1, 3)
    plt.plot(time, yaw, label="Yaw real", color='g')
    if yaw_pred is not None:
        plt.plot(time, yaw_pred, label="Yaw predicted",color='orange')
    plt.xlabel("Time")
    plt.ylabel("Yaw (degrees)")
    plt.title("Yaw")
    plt.legend()

    plt.tight_layout()
    plt.show()
