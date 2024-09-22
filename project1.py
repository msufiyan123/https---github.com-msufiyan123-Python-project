import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# 1. Algorithm Selection: EWMA for anomaly detection
class EWMAAnomalyDetector:
    def __init__(self, alpha=0.3, threshold=3):
        self.alpha = alpha
        self.threshold = threshold
        self.mean = None
        self.var = None
    
    def update(self, value):
        if self.mean is None: 
            self.mean = value
            self.var = 0
        else:
            diff = value - self.mean
            self.mean += self.alpha * diff
            self.var = (1 - self.alpha) * (self.var + self.alpha * diff**2)
    
        if self.var == 0:
            return False
        deviation = np.abs(value - self.mean) / np.sqrt(self.var)
        return deviation > self.threshold

# 2. Data Stream Simulation
def data_stream_generator():
    # Simulate data stream with seasonality and noise
    period = 50
    counter = 0
    while True:
        seasonality = 10 * np.sin(2 * np.pi * counter / period)
        noise = np.random.normal(0, 2)
        if np.random.random() < 0.05:
            # Introduce a random anomaly with 5% probability
            anomaly = np.random.choice([20, -20])
            value = seasonality + noise + anomaly
        else:
            value = seasonality + noise
        yield value
        counter += 1

# 3. Real-time Visualization
def animate(i, detector, data_gen, data, anomaly_points):
    value = next(data_gen)
    data.append(value)
    
    # Anomaly detection
    detector.update(value)
    if detector.is_anomaly(value):
        anomaly_points.append(len(data) - 1)
    
    ax.clear()
    ax.plot(data, label='Data Stream')
    ax.scatter(anomaly_points, [data[i] for i in anomaly_points], color='r', label='Anomalies')
    ax.legend()
    ax.set_title("Real-time Data Stream and Anomaly Detection")
    ax.set_ylim([-30, 30])
    ax.set_xlim([max(0, len(data)-100), len(data)])

# Initialization
data = deque(maxlen=100)
anomaly_points = []
detector = EWMAAnomalyDetector(alpha=0.2, threshold=3)
data_gen = data_stream_generator()

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, animate, fargs=(detector, data_gen, data, anomaly_points), interval=100)
plt.show()


