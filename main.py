import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os

# Create an output directory for saving results
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def generate_data_stream(num_points=1000):
    """
    Generates a synthetic data stream with seasonal patterns, noise, and random anomalies.

    Parameters:
        num_points (int): The total number of data points to generate.

    Returns:
        tuple: A tuple containing the generated data stream (numpy array) and the indices of the anomalies.
    """
    if num_points <= 0:
        raise ValueError("The number of points must be greater than 0.")

    timestamps = np.arange(num_points)
    seasonal_effect = np.sin(2 * np.pi * timestamps / 50)  # Seasonal component
    noise_component = np.random.normal(0, 0.2, num_points)  # Noise
    data_stream = seasonal_effect + noise_component

    # Randomly introduce anomalies into the data stream
    anomaly_indices = random.sample(range(num_points), k=int(0.05 * num_points))
    for idx in anomaly_indices:
        data_stream[idx] += random.uniform(3, 6)  # Introduce spikes

    print("Data stream generation complete.")
    return data_stream, anomaly_indices


def compute_ema(data, window_size):
    """
    Computes the Exponential Moving Average (EMA) for the given data.

    Parameters:
        data (numpy array): The data stream.
        window_size (int): The window size for EMA calculation.

    Returns:
        list: The computed EMA values with NaN for the initial period.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    ema_values = [np.nan] * (window_size - 1)  # Start with NaN for initial values
    initial_ema = np.mean(data[:window_size])  # Initial EMA based on the first window
    ema_values.append(initial_ema)

    alpha = 2 / (window_size + 1)  # Calculate the smoothing factor

    for value in data[window_size:]:
        new_ema = alpha * value + (1 - alpha) * ema_values[-1]
        ema_values.append(new_ema)

    print("EMA computation complete.")
    return ema_values


def calculate_iqr(data):
    """
    Computes the Interquartile Range (IQR) for the given data.

    Parameters:
        data (numpy array): The data stream.

    Returns:
        float: The IQR value.
        float: The first quartile (Q1).
        float: The third quartile (Q3).
    """
    q1 = np.percentile(data, 25)  # First quartile
    q3 = np.percentile(data, 75)  # Third quartile
    iqr_value = q3 - q1  # Interquartile range
    return iqr_value, q1, q3


def identify_anomalies_iqr(data):
    """
    Identifies anomalies in the data stream using the IQR method.

    Parameters:
        data (numpy array): The original data stream.

    Returns:
        list: A list of indices where anomalies were detected.
    """
    if len(data) == 0:
        raise ValueError("Data stream cannot be empty.")

    anomalies = []
    iqr, q1, q3 = calculate_iqr(data)

    lower_bound = q1 - 1.5 * iqr  # Lower bound for outliers
    upper_bound = q3 + 1.5 * iqr  # Upper bound for outliers

    for i in range(len(data)):
        if data[i] < lower_bound or data[i] > upper_bound:  # Check for anomalies
            anomalies.append(i)

    print(f"Identified {len(anomalies)} anomalies using the IQR method.")
    return anomalies


def identify_anomalies_ema(data, ema_values, threshold=3):
    """
    Identifies anomalies in the data stream using the EMA method.

    Parameters:
        data (numpy array): The original data stream.
        ema_values (list): The computed EMA values.
        threshold (float): The Z-score threshold for detecting anomalies.

    Returns:
        list: A list of indices where anomalies were detected.
    """
    if len(data) == 0 or len(ema_values) == 0:
        raise ValueError("Data stream and EMA values cannot be empty.")

    anomalies = []
    residuals = np.abs(data - ema_values)
    mean_residual = np.nanmean(residuals)
    std_residual = np.nanstd(residuals)

    for i in range(len(data)):
        if not np.isnan(ema_values[i]):  # Check for valid EMA values
            z_score = (
                (data[i] - ema_values[i]) / std_residual if std_residual != 0 else 0
            )
            if (
                np.abs(z_score) > threshold
            ):  # Check if the Z-score exceeds the threshold
                anomalies.append(i)

    print(f"Identified {len(anomalies)} anomalies using the EMA method.")
    return anomalies


def create_visualization(data_stream, anomalies, title, filename):
    """
    Creates an animation visualizing the data stream and detected anomalies.

    Parameters:
        data_stream (numpy array): The original data stream.
        anomalies (list): Indices of detected anomalies.
        title (str): Title for the visualization.
        filename (str): Name of the output animation file.
    """
    figure, axes = plt.subplots()

    def animate(frame):
        """Updates the plot for each animation frame."""
        axes.clear()  # Clear previous plot
        axes.plot(
            data_stream[:frame], label="Data Stream", color="blue", alpha=0.6
        )  # Plot the data stream
        current_anomalies = [
            i for i in anomalies if i < frame
        ]  # Get anomalies up to current frame
        axes.scatter(
            current_anomalies,
            [data_stream[i] for i in current_anomalies],
            color="red",
            label="Detected Anomalies",
        )  # Mark anomalies
        axes.set_title(title)  # Set plot title
        axes.set_xlabel("Time")  # X-axis label
        axes.set_ylabel("Value")  # Y-axis label
        axes.legend()  # Show legend
        axes.grid()  # Show grid

    animation = FuncAnimation(
        figure, animate, frames=len(data_stream), interval=100
    )  # Create animation
    animation.save(
        os.path.join(output_dir, filename), writer="pillow", dpi=100
    )  # Save animation
    print(f"Animation saved as {os.path.join(output_dir, filename)}.")


def save_data_plot(data_stream, anomalies, title, filename):
    """
    Saves a plot of the data stream with identified anomalies.

    Parameters:
        data_stream (numpy array): The original data stream.
        anomalies (list): Indices of detected anomalies.
        title (str): Title for the plot.
        filename (str): Name of the output plot file.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        data_stream, label="Data Stream", color="blue", alpha=0.5
    )  # Plot data stream

    if anomalies:  # If there are anomalies, mark them on the plot
        plt.scatter(
            anomalies,
            [data_stream[i] for i in anomalies],
            color="red",
            label="Anomalies",
            marker="o",
        )

    plt.title(title)  # Set plot title
    plt.xlabel("Time")  # Set X-axis label
    plt.ylabel("Value")  # Set Y-axis label
    plt.legend()  # Show legend
    plt.grid()  # Show grid
    plt.savefig(os.path.join(output_dir, filename))  # Save plot
    plt.close()  # Close the plot
    print(f"Plot of data stream with anomalies saved as {filename}.")


if __name__ == "__main__":
    # Generate a data stream with 1000 points
    data_stream, anomaly_indices = generate_data_stream(1000)

    # Identify anomalies using the IQR method
    anomalies_iqr = identify_anomalies_iqr(data_stream)
    save_data_plot(
        data_stream,
        anomalies_iqr,
        "Data Stream with Anomalies (IQR)",
        "data_stream_with_anomalies_iqr.png",
    )
    create_visualization(
        data_stream,
        anomalies_iqr,
        "Anomaly Detection Using IQR",
        "anomaly_detection_iqr.gif",
    )

    # Identify anomalies using the EMA method
    window_size = 50
    ema_values = compute_ema(data_stream, window_size)
    anomalies_ema = identify_anomalies_ema(data_stream, ema_values)
    save_data_plot(
        data_stream,
        anomalies_ema,
        "Data Stream with Anomalies (EMA)",
        "data_stream_with_anomalies_ema.png",
    )
    create_visualization(
        data_stream,
        anomalies_ema,
        "Anomaly Detection Using EMA",
        "anomaly_detection_ema.gif",
    )
