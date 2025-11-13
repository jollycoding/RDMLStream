import math
import numpy as np


class MicroCluster:
    def __init__(self, data, timestamp, re=1, label=-1, radius=-1., lmbda=1e-4):
        self.n = 1  # Number of samples
        self.nl = 0 if label == -1 else 1  # Number of labeled samples
        self.ls = data  # Linear sum
        self.ss = np.square(data)  # Squared sum
        self.t = 0  # Time since last update
        self.re = re  # Reliability
        self.label = label
        self.radius = radius

        self.lmbda = lmbda
        self.espilon = 0.00005
        self.radius_factor = 1.1
        self.label_history = []

        self.mc_id = -1  # Unique micro-cluster ID

    def insert(self, data, timestamp, labeled=False, true_label=None):
        self.n += 1
        self.nl += 1 if labeled else 0
        self.ls += data
        self.ss += np.square(data)
        self.t = 0  # Reset time counter

        self.radius = self.get_radius()

        if true_label is not None:
            self.label_history.append((timestamp, self.label, true_label))

    def update_reliability(self, probability, increase=True):
        # Update reliability based on propagation probability
        if increase:
            self.re += max(1 - self.re, (1 - self.re) * math.pow(math.e, probability - 1))
        else:
            self.re -= (1 - self.re) * math.pow(math.e, probability)

    def update(self):
        # Decay reliability over time
        self.t += 1
        self.re = self.re * math.pow(math.e, - self.lmbda * self.espilon * self.t)
        return self.re

    def get_deviation(self):
        ls_mean = np.sum(np.square(self.ls / self.n))
        ss_mean = np.sum(self.ss / self.n)
        variance = ss_mean - ls_mean
        variance = 1e-6 if variance < 1e-6 else variance
        radius = np.sqrt(variance)  # Standard deviation
        return radius

    def get_center(self):
        return self.ls / self.n

    def get_radius(self):
        if self.n <= 1:
            return self.radius
        # Calculate radius based on standard deviation
        return max(self.radius, self.get_deviation() * self.radius_factor)

    def __str__(self):
        return f"ID={self.mc_id}, n={self.n}, nl={self.nl}, label={self.label}, t={self.t}, re={self.re:.3f}, ra={self.get_radius():.3f}"