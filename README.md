# Efficient-Data-Stream-Anomaly-Detection

Project Overview: Efficient Data Stream Anomaly Detection

In this project, I focused on developing a robust real-time anomaly detection system tailored for continuous data streams. The motivation behind this endeavor was to address critical issues in various domains, such as finance and system monitoring, where identifying anomalies promptly can prevent significant losses and ensure operational integrity.

Key Objectives
Simulating a Data Stream: The first step involved creating a synthetic data stream that closely resembles real-world data. This was achieved by introducing seasonal patterns using sine waves, overlaying them with random noise to mimic fluctuations in genuine datasets. Additionally, I incorporated random anomalies to simulate spikes or drops, which are crucial for testing the detection algorithms.

Implementing Anomaly Detection Algorithms: For anomaly detection, I implemented two methodologies: Exponential Moving Average (EMA) and Interquartile Range (IQR). Both methods provide distinct advantages, and I aimed to compare their effectiveness.

Exponential Moving Average (EMA): This method computes the average of the data points, giving more weight to recent values. The implementation calculates the EMA based on a specified window size, enabling the detection of significant deviations from the expected trend. I established thresholds to identify anomalies, focusing on the Z-score of the residuals.

Interquartile Range (IQR): The IQR method is based on statistical principles, identifying outliers by calculating the first (Q1) and third quartiles (Q3) and defining a range for acceptable values. Anomalies are flagged if they fall outside 1.5 times the IQR from these quartiles, making it a robust technique for handling skewed distributions.

Visualization: To enhance the understanding of the results, I created visualizations that depict both the data stream and the detected anomalies. Utilizing Matplotlib, I developed animations that illustrate the detection process in real time, allowing for a clearer insight into how anomalies are identified over time.

Comparison of Algorithms: By employing both EMA and IQR methods on the same data stream, I aimed to assess their relative performance. This involved analyzing the number of detected anomalies and their positioning within the data stream, providing valuable insights into which method may be more effective under different conditions.

Conclusion
Through this project, I have gained significant insights into the intricacies of anomaly detection in data streams. The combination of EMA and IQR methods not only showcases diverse approaches to detecting anomalies but also highlights the importance of choosing the right method based on the specific characteristics of the data at hand. The visualization tools I developed serve as an essential component in communicating the results effectively, making the findings accessible and actionable.

This project not only equips me with a deeper understanding of anomaly detection techniques but also prepares me for tackling real-world challenges in data analysis and monitoring.

Steps to run ::
    python -m venv cobble_venv
    source cobble_venv/bin/activate (for mac/linux)
    pip install -r requirements.txt
    python main.py
