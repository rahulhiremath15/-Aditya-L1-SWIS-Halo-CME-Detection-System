import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import xgboost as xgb

class CMEModelTrainer:
    """Train machine learning models for CME detection"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def create_training_labels(self, df):
        """Create training labels based on synthetic CME events"""
        print("Creating training labels...")
        
        # Initialize all as normal solar wind (0)
        df['is_cme'] = 0
        
        # Define synthetic CME events for training
        cme_events = [
            ('2024-08-05 12:00:00', '2024-08-05 15:00:00'),
            ('2024-08-12 08:30:00', '2024-08-12 11:45:00'),
            ('2024-08-20 14:15:00', '2024-08-20 17:30:00'),
        ]
        
        for start_str, end_str in cme_events:
            start_time = pd.to_datetime(start_str)
            end_time = pd.to_datetime(end_str)
            
            mask = (df.index >= start_time) & (df.index <= end_time)
            df.loc[mask, 'is_cme'] = 1
            print(f"Labeled CME event: {start_str} to {end_str} ({mask.sum()} points)")
        
        print(f"Total CME points: {df['is_cme'].sum()}")
        print(f"Total normal points: {(df['is_cme'] == 0).sum()}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        print("Preparing features for ML...")
        
        # Select relevant features for CME detection (enhanced with physics-informed features)
        feature_columns = [
            'alpha_proton_ratio',
            'proton_velocity',
            'proton_temperature',
            'alpha_temperature',
            'proton_density',
            'proton_thermal_speed',
            'proton_beta_enhanced',
            'alfven_speed',
            'alfven_mach',
            'dynamic_pressure',
            'temp_ratio',
            'alpha_proton_ratio_rolling_mean',
            'proton_velocity_rolling_mean',
            'proton_temperature_rolling_mean',
            'alfven_mach_rolling_mean',
            'proton_beta_enhanced_rolling_mean',
            'dynamic_pressure_rolling_mean',
            'alpha_proton_ratio_gradient',
            'proton_velocity_gradient',
            'proton_temperature_gradient',
            'alfven_mach_gradient',
            'dynamic_pressure_gradient',
            'alpha_proton_ratio_rate',
            'proton_velocity_rate',
            'proton_temperature_rate',
            'alfven_mach_rate',
            'dynamic_pressure_rate'
        ]
        
        # Keep only columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Extract features and target
        X = df[available_features].copy()
        y = df['is_cme'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        return X, y
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        rf_model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        print(f"Random Forest CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        print("Training SVM model...")
        
        svm_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        svm_model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
        print(f"SVM CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.models['svm'] = svm_model
        return svm_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost classifier for better performance"""
        print("Training XGBoost model...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
        print(f"XGBoost CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest for anomaly detection"""
        print("Training Isolation Forest model...")
        
        # Use only normal solar wind data for training
        iso_model = IsolationForest(
            contamination=0.1,  # Expect 10% outliers
            random_state=42
        )
        
        iso_model.fit(X_train)
        
        self.models['isolation_forest'] = iso_model
        return iso_model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        for model_name, model in self.models.items():
            if model_name == 'isolation_forest':
                # Isolation Forest returns -1 for outliers, 1 for normal
                y_pred = model.predict(X_test)
                y_pred = (y_pred == -1).astype(int)  # Convert to 0/1
            else:
                y_pred = model.predict(X_test)
            
            print(f"\n{model_name.upper()} Results:")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
    
    def get_feature_importance(self, model_name='xgboost'):
        """Get feature importance from trained model"""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return []
    
    def save_models(self, model_dir='models'):
        """Save all trained models"""
        print(f"Saving models to {model_dir}/")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Saved feature columns to {features_path}")
    
    def train_all_models(self, data_path):
        """Complete model training pipeline"""
        print("Starting model training pipeline...")
        
        # Load processed data
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        
        # Create training labels
        df = self.create_training_labels(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_isolation_forest(X_train[y_train == 0])  # Only normal data
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        print("Model training complete!")

# Train models if run directly
if __name__ == "__main__":
    trainer = CMEModelTrainer()
    trainer.train_all_models('data/processed_data/processed_swis_data.csv')