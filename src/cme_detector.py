import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
from .config_loader import config

class CMEEventDetector:
    """Detect CME events using ensemble of methods"""
    
    def __init__(self, model_dir='models'):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.load_models(model_dir)
        self.baseline_stats = {}
    
    def load_models(self, model_dir):
        """Load trained models"""
        print(f"Loading models from {model_dir}/")
        
        try:
            # Load Random Forest
            with open(os.path.join(model_dir, 'random_forest_model.pkl'), 'rb') as f:
                self.models['random_forest'] = pickle.load(f)
            
            # Load SVM
            with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as f:
                self.models['svm'] = pickle.load(f)
            
            # Load XGBoost
            with open(os.path.join(model_dir, 'xgboost_model.pkl'), 'rb') as f:
                self.models['xgboost'] = pickle.load(f)
            
            # Load Isolation Forest
            with open(os.path.join(model_dir, 'isolation_forest_model.pkl'), 'rb') as f:
                self.models['isolation_forest'] = pickle.load(f)
            
            # Load scaler
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature columns
            with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train models first using ml_models.py")
    
    def calculate_statistical_thresholds(self, df):
        """Calculate statistical thresholds for CME detection"""
        print("Calculating statistical thresholds...")
        
        # Calculate baseline statistics
        baseline_stats = {}
        
        for param in ['alpha_proton_ratio', 'proton_velocity', 'proton_temperature']:
            if param in df.columns:
                baseline_stats[param] = {
                    'median': df[param].median(),
                    'mean': df[param].mean(),
                    'std': df[param].std(),
                    'q90': df[param].quantile(0.90),
                    'q95': df[param].quantile(0.95)
                }
        
        self.baseline_stats = baseline_stats
        
        # Define thresholds based on literature and statistics
        thresholds = {
            'alpha_proton_ratio': {
                'enhancement_factor': 2.0,  # 2x baseline
                'absolute_threshold': 0.08   # Literature value
            },
            'proton_velocity': {
                'high_speed': 600,  # km/s for fast CMEs
                'gradient_threshold': 50  # km/s per hour
            },
            'proton_temperature': {
                'enhancement_factor': 1.5,  # 1.5x baseline
                'absolute_threshold': 100000  # 100,000 K
            }
        }
        
        return thresholds
    
    def apply_statistical_detection(self, df, thresholds):
        """Apply statistical threshold detection"""
        print("Applying statistical detection...")
        
        df['stat_detection'] = 0
        
        # Alpha-proton ratio enhancement
        if 'alpha_proton_ratio' in df.columns:
            baseline_ratio = self.baseline_stats.get('alpha_proton_ratio', {}).get('median', None)
            enhancement_factor = thresholds.get('alpha_proton_ratio', {}).get('enhancement_factor', None)
            if baseline_ratio is not None and enhancement_factor is not None:
                df['alpha_proton_ratio'] = pd.to_numeric(df['alpha_proton_ratio'], errors='coerce')
                if 'stat_detection' not in df.columns:
                    df['stat_detection'] = 0
                ratio_condition = df['alpha_proton_ratio'] > (baseline_ratio * enhancement_factor)
                df.loc[ratio_condition, 'stat_detection'] = 1

        
        # High proton temperature
        if 'proton_temperature' in df.columns:
            temp_condition = (df['proton_temperature'] > 
                            thresholds['proton_temperature']['absolute_threshold'])
            df.loc[temp_condition, 'stat_detection'] = 1
        
        # High velocity
        if 'proton_velocity' in df.columns:
            vel_condition = (df['proton_velocity'] > 
                           thresholds['proton_velocity']['high_speed'])
            df.loc[vel_condition, 'stat_detection'] = 1
        
        return df
    
    def adaptive_ensemble_weights(self, df):
        """Dynamic weight adjustment based on data conditions"""
        # Calculate current solar wind conditions
        current_velocity = df['proton_velocity'].rolling(window=60).mean().iloc[-1] if len(df) > 60 else df['proton_velocity'].mean()
        current_density = df['proton_density'].rolling(window=60).mean().iloc[-1] if len(df) > 60 else df['proton_density'].mean()
        
        # Adaptive weights based on conditions
        if current_velocity > 500:  # High speed stream
            weights = {
                'statistical': 0.25, 
                'rf': 0.25, 
                'svm': 0.20, 
                'iso': 0.10,
                'xgb': 0.20  # Add XGBoost
            }
        elif current_density > 10:  # High density
            weights = {
                'statistical': 0.30, 
                'rf': 0.25, 
                'svm': 0.20, 
                'iso': 0.10,
                'xgb': 0.15
            }
        else:  # Quiet solar wind
            weights = {
                'statistical': 0.35, 
                'rf': 0.25, 
                'svm': 0.15, 
                'iso': 0.10,
                'xgb': 0.15
            }
        
        return weights
    
    def apply_ml_detection(self, df):
        """Apply machine learning detection"""
        print("Applying ML detection...")
        
        # Safety check for models and scaler
        if not self.models or self.scaler is None:
            print("ML models/scaler not loaded — skipping ML detection.")
            df['rf_probability'] = 0.0
            df['svm_probability'] = 0.0
            df['xgb_probability'] = 0.0
            df['iso_prediction'] = 0
            df['rf_prediction'] = 0
            df['svm_prediction'] = 0
            df['xgb_prediction'] = 0
            return df
        
        # Prepare features
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict(X_scaled)
        rf_prob = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
        # SVM prediction
        svm_pred = self.models['svm'].predict(X_scaled)
        svm_prob = self.models['svm'].predict_proba(X_scaled)[:, 1]
        
        # XGBoost prediction
        xgb_pred = self.models['xgboost'].predict(X_scaled)
        xgb_prob = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
        
        # Isolation Forest prediction (anomaly detection)
        iso_pred = self.models['isolation_forest'].predict(X_scaled)
        iso_pred = (iso_pred == -1).astype(int)  # Convert to 0/1
        
        # Add predictions to dataframe
        df['rf_prediction'] = rf_pred
        df['rf_probability'] = rf_prob
        df['svm_prediction'] = svm_pred
        df['svm_probability'] = svm_prob
        df['xgb_prediction'] = xgb_pred
        df['xgb_probability'] = xgb_prob
        df['iso_prediction'] = iso_pred
        
        return df
    
    def ensemble_detection(self, df):
        """Combine all detection methods with adaptive weights"""
        print("Applying ensemble detection...")
        
        # Get adaptive weights
        weights = self.adaptive_ensemble_weights(df)
        
        # Weighted ensemble with dynamic weights
        df['ensemble_score'] = (
            weights['statistical'] * df['stat_detection'] +
            weights['rf'] * df['rf_probability'] +
            weights['svm'] * df['svm_probability'] +
            weights['xgb'] * df['xgb_probability'] +
            weights['iso'] * df['iso_prediction']
        )
        
        # Final detection threshold
        df['is_cme_detected'] = (df['ensemble_score'] > 0.5).astype(int)
        
        return df
    
    def extract_events(self, df, min_duration_minutes=30):
        """Extract individual CME events from detection time series"""
        print("Extracting individual events...")
        
        events = []
        cme_mask = df['is_cme_detected'] == 1
        
        if not cme_mask.any():
            print("No CME events detected.")
            return []
        
        # Find contiguous blocks of CME detections
        cme_indices = df.index[cme_mask]
        
        if len(cme_indices) == 0:
            return []
        
        # Group consecutive timestamps
        event_start = cme_indices[0]
        event_end = cme_indices[0]
        
        for i in range(1, len(cme_indices)):
            current_time = cme_indices[i]
            
            # If gap is more than 30 minutes, start new event
            if (current_time - event_end).total_seconds() > 30 * 60:
                # Save previous event if long enough
                duration = (event_end - event_start).total_seconds() / 60
                if duration >= min_duration_minutes:
                    events.append(self.characterize_event(df, event_start, event_end))
                
                # Start new event
                event_start = current_time
            
            event_end = current_time
        
        # Don't forget the last event
        duration = (event_end - event_start).total_seconds() / 60
        if duration >= min_duration_minutes:
            events.append(self.characterize_event(df, event_start, event_end))
        
        print(f"Extracted {len(events)} CME events")
        return events
    
    def characterize_event(self, df, start_time, end_time):
        """Characterize a single CME event"""
        event_data = df[start_time:end_time]
        
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Calculate event characteristics
        event_info = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_hours': round(duration_hours, 2),
            'max_alpha_proton_ratio': event_data['alpha_proton_ratio'].max(),
            'max_velocity': event_data['proton_velocity'].max(),
            'max_temperature': event_data['proton_temperature'].max(),
            'mean_ensemble_score': event_data['ensemble_score'].mean(),
            'confidence': 'High' if event_data['ensemble_score'].mean() > 0.7 else 
                         'Medium' if event_data['ensemble_score'].mean() > 0.5 else 'Low'
        }
        
        # Classify event type based on velocity
        max_vel = event_info['max_velocity']
        if max_vel > 600:
            event_info['event_type'] = 'Fast CME'
        elif max_vel < 350:
            event_info['event_type'] = 'Slow CME'
        else:
            event_info['event_type'] = 'Moderate CME'
        
        return event_info
    
    def detect_cme_events(self, df):
        """Complete CME detection pipeline"""
        print("Starting CME detection pipeline...")
        
        # Calculate statistical thresholds
        thresholds = self.calculate_statistical_thresholds(df)
        
        # Apply statistical detection
        df = self.apply_statistical_detection(df, thresholds)
        
        # Apply ML detection
        df = self.apply_ml_detection(df)
        
        # Apply ensemble detection
        df = self.ensemble_detection(df)
        
        # Extract individual events
        events = self.extract_events(df)
        
        results = {
            'full_data_with_detections': df,
            'events': events,
            'thresholds': thresholds,
            'baseline_stats': self.baseline_stats,
            'summary': {
                'total_events': len(events),
                'total_detection_points': df['is_cme_detected'].sum(),
                'detection_rate': df['is_cme_detected'].mean()
            }
        }
        
        print("CME detection complete!")
        print(f"Total events detected: {len(events)}")
        
        return results
 # Test the detector
if __name__ == "__main__":
    detector = CMEEventDetector()
    
    # Load processed data
    df = pd.read_csv('data/processed_data/processed_swis_data.csv', 
                     index_col='timestamp', parse_dates=True)
    
    # Run detection
    results = detector.detect_cme_events(df)
    
    # Print results
    for i, event in enumerate(results['events']):
        print(f"\nEvent {i+1}:")
        for key, value in event.items():
            print(f"  {key}: {value}")
