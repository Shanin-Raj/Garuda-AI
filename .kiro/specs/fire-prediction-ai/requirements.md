# Requirements Document: Fire Prediction AI System

## Introduction

The Fire Prediction AI System is a drone-based first response network that predicts fire intensity and spreading patterns using multiple proven measurement techniques. The system integrates aerial sensor data (MWIR, TIR, LiDAR, optical/NIR imagery) with machine learning models to provide real-time fire analysis and spreading predictions for emergency response teams.

## Glossary

- **System**: The Fire Prediction AI System
- **FRP**: Fire Radiative Power - radiant heat energy measured in megawatts (MW)
- **FRE**: Fire Radiative Energy - cumulative radiant energy over time
- **MWIR**: Mid-Wave Infrared (3–5 μm wavelength)
- **TIR**: Thermal Infrared
- **Byram_Intensity**: Fireline intensity calculated as I_B = H × w × ROS (kW/m)
- **ROS**: Rate of Spread - speed at which fire advances
- **FRED**: Fire Radiative Energy Density
- **NBR**: Normalized Burn Ratio - spectral index for burned area identification (-1 to 1)
- **dNBR**: Differenced Normalized Burn Ratio - pre-fire vs post-fire difference
- **RdNBR**: Relative dNBR - normalized by pre-fire conditions
- **LiDAR**: Light Detection and Ranging - laser-based distance measurement
- **NIR**: Near-Infrared spectrum
- **Drone_Network**: Network of aerial drones collecting sensor data
- **Analysis_Report**: Detailed output document containing all measurements and predictions
- **Fire_Perimeter**: Boundary of active fire area
- **Fuel_Load**: Amount of combustible vegetation per unit area
- **Heat_of_Combustion**: Energy released per unit mass of fuel (H)
- **Severity_Threshold**: Classification boundary for low/moderate/high burn severity

## Requirements

### Requirement 1: Fire Radiative Power and Energy Measurement

**User Story:** As a fire analyst, I want to measure fire radiative power and energy using MWIR sensors, so that I can quantify the radiant heat energy and identify hot spots.

#### Acceptance Criteria

1. WHEN MWIR sensor data (3–5 μm) is received, THE System SHALL calculate Fire Radiative Power in megawatts
2. WHEN temperature-dependent emission data is available, THE System SHALL compute cumulative Fire Radiative Energy
3. WHEN hot spots are detected in thermal imagery, THE System SHALL identify and catalog their locations within the Fire_Perimeter
4. THE System SHALL process MWIR data with spatial resolution sufficient to detect individual fire fronts

### Requirement 2: Byram's Fireline Intensity Calculation

**User Story:** As a fire behavior specialist, I want to calculate Byram's fireline intensity using the formula I_B = H × w × ROS, so that I can assess fire behavior and suppression difficulty.

#### Acceptance Criteria

1. WHEN high-resolution airborne infrared video is received, THE System SHALL calculate Rate of Spread using pixel-based motion analysis
2. WHEN pre-burn and post-burn LiDAR data are available, THE System SHALL determine fuel consumed per unit area (w)
3. WHEN Heat_of_Combustion, fuel consumption, and ROS are known, THE System SHALL compute Byram_Intensity in kW/m
4. THE System SHALL apply the FRED-ROS method to map Fire Radiative Energy Density across the Fire_Perimeter
5. WHEN Byram_Intensity exceeds 4000 kW/m, THE System SHALL flag the fire as high-intensity requiring specialized suppression

### Requirement 3: Spectral Index Analysis

**User Story:** As a burn severity assessor, I want to calculate NBR, dNBR, and RdNBR indices, so that I can identify burned areas and determine fire severity.

#### Acceptance Criteria

1. WHEN optical and NIR imagery are received, THE System SHALL calculate Normalized Burn Ratio with values ranging from -1 to 1
2. WHEN pre-fire and post-fire NBR values are available, THE System SHALL compute differenced NBR (dNBR)
3. WHEN dNBR and pre-fire NBR are known, THE System SHALL calculate Relative dNBR normalized by pre-fire conditions
4. THE System SHALL classify burn severity as low, moderate, or high based on Severity_Threshold values
5. THE System SHALL verify that NBR is negatively correlated with fire intensity in the analysis

### Requirement 4: Machine Learning Fire Classification

**User Story:** As a fire prediction operator, I want the system to use machine learning models for fire intensity classification, so that I can leverage automated pattern recognition.

#### Acceptance Criteria

1. WHEN multi-sensor data (LiDAR, IR, visible imagery) is collected, THE System SHALL integrate all sensor inputs into a unified feature set
2. THE System SHALL apply YOLO-like models to classify fire intensity into discrete categories
3. WHEN individual tree data is available, THE System SHALL use Random Forest models to predict tree mortality with minimum 75% accuracy
4. THE System SHALL retrain models periodically using validated field data to maintain prediction accuracy

### Requirement 5: Fire Spreading Prediction

**User Story:** As an emergency response coordinator, I want predictions of fire spreading rate and direction, so that I can plan evacuations and resource deployment.

#### Acceptance Criteria

1. WHEN current vegetation data for the terrain is available, THE System SHALL incorporate month-specific fuel characteristics into spreading models
2. WHEN wind speed and direction predictions are received, THE System SHALL factor wind vectors into spreading calculations
3. WHEN weather pattern forecasts for 2 days from incident are available, THE System SHALL project fire behavior under predicted conditions
4. THE System SHALL calculate possible spreading rate in meters per hour
5. THE System SHALL determine spreading direction as compass bearings with confidence intervals
6. WHEN terrain slope data is available, THE System SHALL adjust spreading predictions for topographic effects

### Requirement 6: Multi-Sensor Data Integration

**User Story:** As a system operator, I want seamless integration of data from multiple drone sensors, so that I can obtain comprehensive fire measurements.

#### Acceptance Criteria

1. WHEN the Drone_Network transmits sensor data, THE System SHALL accept FRP/FRE measurements from MWIR/TIR sensors
2. WHEN infrared video and LiDAR data are received, THE System SHALL process them for Byram_Intensity calculations
3. WHEN optical and NIR imagery arrive, THE System SHALL compute spectral indices (NBR, dNBR, RdNBR)
4. THE System SHALL synchronize timestamps across all sensor inputs to ensure temporal alignment
5. WHEN sensor data quality is insufficient, THE System SHALL flag unreliable measurements and request additional data collection

### Requirement 7: Analysis Report Generation

**User Story:** As a fire incident commander, I want detailed analysis reports with all measurements and predictions, so that I can make informed tactical decisions.

#### Acceptance Criteria

1. WHEN all measurements are complete, THE System SHALL generate an Analysis_Report containing fire intensity measurements
2. THE Analysis_Report SHALL include possible spreading rate with uncertainty bounds
3. THE Analysis_Report SHALL specify spreading direction with visual map overlays
4. THE Analysis_Report SHALL contain detailed analysis of all input measurements (FRP, FRE, Byram_Intensity, spectral indices)
5. THE Analysis_Report SHALL include confidence scores for all predictions
6. WHEN tree mortality predictions are available, THE Analysis_Report SHALL include spatial maps of predicted mortality
7. THE System SHALL deliver the Analysis_Report within 5 minutes of data collection completion

### Requirement 8: Real-Time Processing

**User Story:** As a first responder, I want near-real-time fire analysis, so that I can react quickly to changing fire conditions.

#### Acceptance Criteria

1. WHEN new sensor data arrives from the Drone_Network, THE System SHALL begin processing within 10 seconds
2. THE System SHALL update fire intensity measurements continuously as new data streams in
3. WHEN significant changes in fire behavior are detected, THE System SHALL issue alerts within 30 seconds
4. THE System SHALL maintain processing throughput sufficient for simultaneous monitoring of multiple fire incidents

### Requirement 9: Data Validation and Quality Control

**User Story:** As a system administrator, I want automated data validation, so that I can ensure prediction accuracy and reliability.

#### Acceptance Criteria

1. WHEN sensor data is received, THE System SHALL validate data format and completeness before processing
2. WHEN calculated values fall outside physically plausible ranges, THE System SHALL flag anomalies for review
3. THE System SHALL cross-validate measurements from multiple sensors to detect inconsistencies
4. WHEN validation fails, THE System SHALL log errors with specific sensor and timestamp information
5. THE System SHALL maintain a quality score for each Analysis_Report based on input data quality

### Requirement 10: Historical Data Storage and Retrieval

**User Story:** As a fire researcher, I want access to historical fire data and predictions, so that I can validate models and improve future predictions.

#### Acceptance Criteria

1. THE System SHALL store all sensor inputs, calculated measurements, and predictions with associated metadata
2. WHEN a user queries historical data, THE System SHALL retrieve records by date, location, or fire incident identifier
3. THE System SHALL preserve the complete Analysis_Report for each fire incident for minimum 5 years
4. WHEN model retraining occurs, THE System SHALL use historical data to validate prediction improvements
