# HMM-Based Human Activity Recognition

A complete implementation of Hidden Markov Model (HMM) for recognizing human activities from smartphone sensor data.

## Overview

This project uses accelerometer and gyroscope data collected from a smartphone to classify four activities:
- Standing
- Walking
- Jumping
- Still

The system extracts time-domain and frequency-domain features from sensor windows, trains an HMM with Gaussian emission probabilities, and uses the Viterbi algorithm for activity decoding.

## Features

- **Data Processing**: Loads and preprocesses sensor data from CSV files
- **Feature Extraction**: Computes 15+ features including:
  - Time-domain: mean, std, variance, SMA, correlations
  - Frequency-domain: dominant frequency, spectral energy
- **HMM Implementation**: Custom HMM with:
  - Gaussian emission probabilities
  - Viterbi decoding algorithm
  - Transition matrix learning
- **Evaluation**: Comprehensive metrics including:
  - Sensitivity and specificity per activity
  - Confusion matrix
  - Overall accuracy
- **Visualizations**: 
  - Time series plots
  - Feature correlation heatmap
  - Transition matrix heatmap
  - Confusion matrix
  - Decoded activity timeline




### Data Format

Each CSV file should contain:
- `timestamp`: Unix timestamp
- `acc_x`, `acc_y`, `acc_z`: Accelerometer readings (m/s²)
- `gyr_x`, `gyr_y`, `gyr_z`: Gyroscope readings (rad/s)
- `activity`: Activity label (standing, walking, jumping, still)

Sampling rate: ~100 Hz

## Results

The analysis generates:
- **Visualizations**: PNG images in `public/results/`
- **Metrics**: JSON file with detailed performance metrics
- **Console Output**: Step-by-step progress and summary statistics

## Technical Details

### Feature Extraction
- Window size: 2 seconds (200 samples at 100 Hz)
- Overlap: 50%
- Features: 15 total (time + frequency domain)

### HMM Model
- States: 4 (one per activity)
- Emission model: Multivariate Gaussian
- Decoding: Viterbi algorithm
- Training: Maximum likelihood estimation with transition refinement

### Evaluation Metrics
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Overall Accuracy
- Confusion Matrix

## Dependencies

Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

## Future Improvements

- Add magnetometer data for orientation invariance
- Implement Baum-Welch algorithm for parameter refinement
- Explore deep learning approaches (LSTM, Transformer)
- Add real-time activity recognition
- Expand to more activity types
