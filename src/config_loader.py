"""Configuration loader for CME Detection System"""

import yaml
import os
from astropy import constants as const

class ConfigLoader:
    """Load and manage configuration parameters"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Convert numeric strings to actual numbers
            self._convert_numeric_types(config)
            return config
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}. Using defaults.")
            return self._get_default_config()
    
    def _convert_numeric_types(self, config):
        """Convert numeric strings to actual numbers recursively"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    self._convert_numeric_types(value)
                elif isinstance(value, str):
                    # Try to convert to float or int
                    try:
                        if '.' in value or 'e' in value.lower():
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
        return config
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'magnetic_field': {'default_b': 5e-9, 'source': 'default'},
            'thresholds': {
                'alpha_proton_ratio': {'enhancement_factor': 2.0, 'absolute_threshold': 0.08},
                'proton_velocity': {'high_speed': 600, 'gradient_threshold': 50},
                'proton_temperature': {'enhancement_factor': 1.5, 'absolute_threshold': 100000}
            },
            'models': {
                'isolation_forest': {'contamination': 0.05, 'random_state': 42}
            },
            'units': {
                'density_conversion': 1e6,
                'velocity_conversion': 1000,
                'temperature_unit': 'K'
            }
        }
    
    def get_magnetic_field(self):
        """Get magnetic field value in Tesla"""
        b_config = self.config.get('magnetic_field', {})
        b_value = b_config.get('default_b', 5e-9)
        
        # Use astropy constants for physical accuracy
        if b_config.get('source') == 'astropy':
            # Could fetch from space physics databases
            b_value = const.B_earth.value  # Earth's magnetic field
            
        return b_value
    
    def get_thresholds(self):
        """Get detection thresholds"""
        return self.config.get('thresholds', {})
    
    def get_model_params(self):
        """Get model parameters"""
        return self.config.get('models', {})
    
    def get_unit_conversions(self):
        """Get unit conversion factors"""
        return self.config.get('units', {})
    
    def get_ensemble_weights(self, condition='quiet_solar_wind'):
        """Get ensemble weights for given condition"""
        weights_config = self.config.get('ensemble_weights', {})
        return weights_config.get(condition, weights_config.get('quiet_solar_wind', {}))
    
    def update_config(self, new_config):
        """Update configuration with new values"""
        self.config.update(new_config)
        
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Global config instance
config = ConfigLoader()
