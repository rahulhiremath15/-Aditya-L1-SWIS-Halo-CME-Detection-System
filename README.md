# 🌞 Aditya-L1 SWIS Halo CME Detection System

>Advanced Machine Learning for Space Weather Monitoring

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/rahulhiremath15/-Aditya-L1-SWIS-Halo-CME-Detection-System)

## 🎯 Overview

This system detects **Coronal Mass Ejections (CMEs)** using data from India's **Aditya-L1 SWIS (Solar Wind Ion Spectrometer)** instrument. It combines state-of-the-art machine learning with physics-based approaches to provide accurate, real-time CME detection for space weather monitoring.

### 🌟 Key Features

- **🤖 Advanced ML Ensemble**: XGBoost + Random Forest + SVM + Isolation Forest
- **⚡ Adaptive Weighting**: Context-aware detection based on solar wind conditions  
- **🔬 Physics-Informed**: Alfvén speed, plasma beta, dynamic pressure calculations
- **📊 Real-time Processing**: Handles minute-resolution solar wind data
- **🌐 Web Dashboard**: Interactive visualization and monitoring interface
- **📈 High Accuracy**: 85-95% detection rate with <0.5% false positives

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (tested on 3.13.5)
- **8GB+ RAM** recommended for optimal performance
- **Windows/Linux/macOS** support

### Installation

```bash
# Clone the repository
git clone https://github.com/rahulhiremath15/-Aditya-L1-SWIS-Halo-CME-Detection-System
cd -Aditya-L1-SWIS-Halo-CME-Detection-System

# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python main.py --train

# Run detection with sample data
python main.py --sample --visualize

# Launch web dashboard
python main.py --web
```

## 📋 Usage Guide

### 1. **Training the Models**

```bash
python main.py --train
```

**Output**: Trained models saved to `models/` directory
- Random Forest, SVM, XGBoost, Isolation Forest
- Feature scaler and column mappings
- Cross-validation scores and performance metrics

### 2. **Running CME Detection**

#### **With Sample Data**
```bash
python main.py --sample --visualize
```

#### **With Custom Data**
```bash
python main.py --file path/to/your/swis_data.csv --visualize
```

**Expected Data Format** (CSV):
```csv
timestamp,proton_flux,alpha_flux,proton_density,proton_velocity,proton_temperature,alpha_temperature
2024-08-01 00:00:00,2.1e8,8.2e6,7.2,398,52000,105000
2024-08-01 00:01:00,2.0e8,7.9e6,6.8,402,51000,98000
...
```

### 3. **Web Dashboard**

```bash
python main.py --web
```

Access at: **http://localhost:5000**

Features:
- 📊 Real-time CME event monitoring
- 📈 Detection statistics and summaries  
- 🎯 Interactive event details
- 📥 Download detection reports

## 📊 Understanding Results

### **Output Files**

#### **Detection Report** (`results/reports/cme_detection_report.json`)
```json
{
  "summary": {
    "total_events": 3,
    "detection_rate": 0.0033,
    "total_detection_points": 130
  },
  "detected_events": [
    {
      "start_time": "2024-08-05 12:01:00",
      "end_time": "2024-08-05 14:59:00", 
      "duration_hours": 2.97,
      "max_velocity": 808,
      "max_temperature": 148612,
      "max_alpha_proton_ratio": 0.489,
      "confidence": "High",
      "event_type": "Fast CME"
    }
  ]
}
```

#### **Visualizations** (`results/plots/`)
- `full_timeseries_with_events.png` - Complete solar wind overview
- `event_1_detail.png` - Individual CME event analysis
- `statistics_summary.png` - Detection statistics and distributions

### **Interpreting CME Events**

| Parameter | Normal Range | CME Indication |
|-----------|---------------|-----------------|
| **α/p Ratio** | 0.03-0.05 | >0.08 (2x+ enhancement) |
| **Velocity** | 300-500 km/s | >600 km/s (Fast CME) |
| **Temperature** | 30K-100K K | >100K K (1.5x+ enhancement) |
| **Confidence** | - | High/Medium/Low based on ensemble score |

## 🔧 Advanced Configuration

### **Custom Detection Thresholds**

Edit `src/cme_detector.py` to modify:

```python
thresholds = {
    'alpha_proton_ratio': {
        'enhancement_factor': 2.0,  # Adjust sensitivity
        'absolute_threshold': 0.08   # Minimum threshold
    },
    'proton_velocity': {
        'high_speed': 600,  # Fast CME threshold
        'gradient_threshold': 50  # Rate of change threshold
    }
}
```

### **Model Retraining**

```bash
# Add new labeled CME events to ml_models.py
python src/ml_models.py  # Direct model training
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data    │───▶│  Data Processor │───▶│ CME Detector  │
│  (SWIS L2)    │    │  - Physics     │    │  - ML Ensemble │
│                │    │  - Features    │    │  - Adaptive    │
└─────────────────┘    └──────────────────┘    │  - Events      │
                                              └─────────────────┘
                                                       │
                                              ▼
                                        ┌─────────────────┐
                                        │  Visualizer    │
                                        │  - Plots       │
                                        │  - Reports      │
                                        └─────────────────┘
```

### **Machine Learning Pipeline**

1. **Feature Engineering** (53 parameters)
   - Solar wind basics: velocity, density, temperature
   - Physics-informed: Alfvén speed, plasma beta, dynamic pressure  
   - Statistical: rolling means, gradients, rates of change

2. **Ensemble Detection**
   - **XGBoost**: Primary classifier (highest accuracy)
   - **Random Forest**: Complementary decision trees
   - **SVM**: Support vector machine for complex boundaries
   - **Isolation Forest**: Anomaly detection for unusual events

3. **Adaptive Weighting**
   - **High-speed streams**: Emphasize ML models
   - **High-density periods**: Balanced approach
   - **Quiet solar wind**: Emphasize statistical thresholds

## 📈 Performance Metrics

### **Validation Results**

| Model | Cross-Validation | F1-Score | Precision |
|--------|------------------|------------|-----------|
| **XGBoost** | 100% | 96% | 96% |
| **Random Forest** | 99.9% | 90% | 84% |
| **SVM** | 99.9% | 85% | 85% |
| **Ensemble** | - | **92%** | **89%** |

### **Historical Validation**
- **✅ August 2024 CMEs**: 67% correlation with NOAA records
- **✅ Physical parameters**: Within expected CME ranges
- **✅ False positive rate**: 0.33% (realistic for space weather)

## 🌍 Deployment

### **Production Setup**

```bash
# Set up production environment
python -m venv cme_production
source cme_production/bin/activate  # Linux/macOS
# cme_production\Scripts\activate  # Windows

pip install -r requirements.txt

# Train with historical data
python main.py --train

# Set up monitoring service
python main.py --web --monitor
```

### **Docker Deployment**

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py", "--web"]
```

```bash
docker build -t -Aditya-L1-SWIS-Halo-CME-Detection-System .
docker run -p 5000:5000 -Aditya-L1-SWIS-Halo-CME-Detection-System
```

## 🔬 Scientific Background

### **CME Detection Principles**

1. **Alpha-Proton Ratio Enhancement**
   - Normal solar wind: α/p ≈ 0.04
   - CME signature: α/p > 0.08 (2x+ enhancement)
   - Indicates heavy ion enrichment from solar corona

2. **Velocity Increases**
   - Background: 300-500 km/s
   - Fast CMEs: >600 km/s
   - Associated with geomagnetic storm potential

3. **Temperature Enhancement**
   - Normal: 30K-100K K
   - CME: >100K K (1.5x+ increase)
   - Reflects heating in coronal mass ejection

### **Aditya-L1 SWIS Instrument**

- **Orbit**: L1 Lagrange Point (1.5M km from Earth)
- **Coverage**: Continuous solar wind monitoring
- **Parameters**: Ion composition, velocity, temperature
- **Advantage**: ~45 minute warning before Earth impact

## 🤝 Contributing

### **Development Setup**

```bash
# Clone repository
git clone https://github.com/rahulhiremath15/-Aditya-L1-SWIS-Halo-CME-Detection-System.git
cd -Aditya-L1-SWIS-Halo-CME-Detection-System

# Create feature branch
git checkout -b feature/your-improvement

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/
```

### **Adding New Features**

1. **New Models**: Add to `src/ml_models.py`
2. **New Features**: Add to `src/data_processor.py`
3. **New Visualizations**: Add to `src/visualizer.py`
4. **Update Documentation**: Modify this README

## 🙏 Acknowledgments

- **ISRO** - Aditya-L1 mission team and SWIS instrument data
- **NOAA SWPC** - Historical CME validation data
