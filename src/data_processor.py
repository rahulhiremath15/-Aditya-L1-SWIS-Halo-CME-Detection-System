import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class SWISDataProcessor:
    """Process SWIS Level-2 data for CME detection"""
    
    def __init__(self):
        self.processed_data = None
        self.baseline_stats = {}
    
    def load_data(self, file_path):
        """Load SWIS data from CSV file"""
        print(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and quality control SWIS data"""
        print("Cleaning data...")
        
        # Remove non-physical values
        df = df[(df['proton_flux'] > 0) & (df['alpha_flux'] > 0)]
        df = df[(df['proton_velocity'] > 200) & (df['proton_velocity'] < 1000)]
        df = df[(df['proton_temperature'] > 1000) & (df['proton_temperature'] < 1e6)]
        
        # Remove extreme outliers (beyond 5 sigma)
        for col in ['proton_flux', 'alpha_flux', 'proton_density', 
                   'proton_velocity', 'proton_temperature']:
            if col in df.columns:
                z_scores = np.abs(zscore(df[col]))
                df = df[z_scores < 5]
        
        print(f"Data cleaned. Shape after cleaning: {df.shape}")
        return df
    
    def calculate_derived_parameters(self, df):
        """Calculate derived parameters for CME detection"""
        print("Calculating derived parameters...")
        
        # Alpha-to-proton ratio (key CME indicator)
        df['alpha_proton_ratio'] = df['alpha_flux'] / df['proton_flux']
        
        # Thermal speeds
        mp = 1.67e-27  # proton mass in kg
        k = 1.38e-23   # Boltzmann constant
        df['proton_thermal_speed'] = np.sqrt(2 * k * df['proton_temperature'] / mp) / 1000  # km/s
        
        # Enhanced plasma beta calculation
        B = 5e-9  # Typical IMF strength in Tesla (can be made configurable)
        mu_0 = 4 * np.pi * 1e-7
        df['proton_beta_enhanced'] = (2 * 1.67e-27 * df['proton_density'] * 1e6 * k * df['proton_temperature']) / (B**2 / mu_0)
        
        # Alfvén speed calculation
        df['alfven_speed'] = B / np.sqrt(mu_0 * df['proton_density'] * 1.67e-27 * 1e6) / 1000  # km/s
        
        # Alfvén Mach number
        df['alfven_mach'] = df['proton_velocity'] / df['alfven_speed']
        
        # Temperature ratio
        df['temp_ratio'] = df['alpha_temperature'] / df['proton_temperature']
        
        # Dynamic pressure
        df['dynamic_pressure'] = df['proton_density'] * 1.67e-27 * 1e6 * (df['proton_velocity'] * 1000)**2  # Pa
        
        return df
    
    def calculate_rolling_statistics(self, df, window_hours=12):
        """Calculate rolling statistics for baseline determination with vectorization"""
        print(f"Calculating {window_hours}-hour rolling statistics...")
        
        window_size = window_hours * 60  # Convert hours to minutes
        
        # Vectorized rolling statistics for key parameters
        key_params = ['alpha_proton_ratio', 'proton_velocity', 'proton_temperature', 
                      'alfven_mach', 'proton_beta_enhanced', 'dynamic_pressure']
        
        for param in key_params:
            if param in df.columns:
                # Use pandas rolling with efficient calculations
                rolling = df[param].rolling(window=window_size, center=True, min_periods=1)
                df[f'{param}_rolling_mean'] = rolling.mean()
                df[f'{param}_rolling_std'] = rolling.std()
                df[f'{param}_rolling_median'] = rolling.median()
                df[f'{param}_rolling_max'] = rolling.max()
                df[f'{param}_rolling_min'] = rolling.min()
        
        return df
    
    def calculate_gradients(self, df):
        """Calculate time gradients for change detection with vectorization"""
        print("Calculating temporal gradients...")
        
        # Vectorized gradient calculations
        gradient_params = ['proton_velocity', 'proton_temperature', 'alpha_proton_ratio',
                         'alfven_mach', 'dynamic_pressure']
        
        for param in gradient_params:
            if param in df.columns:
                # Use numpy gradient for efficiency
                values = df[param].values
                df[f'{param}_gradient'] = np.gradient(values)
                # Also calculate rate of change (second derivative)
                df[f'{param}_rate'] = np.gradient(np.gradient(values))
        
        return df
    
    def calculate_baseline_statistics(self, df):
        """Calculate baseline statistics for threshold determination"""
        print("Calculating baseline statistics...")
        
        # Use rolling median as baseline (less sensitive to outliers)
        baseline_stats = {}
        
        # Enhanced parameter set for baseline
        enhanced_params = ['alpha_proton_ratio', 'proton_velocity', 'proton_temperature',
                          'alfven_mach', 'proton_beta_enhanced', 'dynamic_pressure']
        
        for param in enhanced_params:
            if param in df.columns:
                baseline_stats[param] = {
                    'median': df[param].median(),
                    'mean': df[param].mean(),
                    'std': df[param].std(),
                    'q75': df[param].quantile(0.75),
                    'q90': df[param].quantile(0.90),
                    'q95': df[param].quantile(0.95),
                    'q99': df[param].quantile(0.99)
                }
        
        self.baseline_stats = baseline_stats
        return baseline_stats
    
    def process_swis_data(self, file_path):
        """Complete data processing pipeline"""
        print("Starting SWIS data processing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Clean data
        df = self.clean_data(df)
        
        # Calculate derived parameters
        df = self.calculate_derived_parameters(df)
        
        # Calculate rolling statistics
        df = self.calculate_rolling_statistics(df)
        
        # Calculate gradients
        df = self.calculate_gradients(df)
        
        # Calculate baseline statistics
        baseline_stats = self.calculate_baseline_statistics(df)
        
        # Drop rows with NaN values from rolling calculations
        df = df.dropna()
        
        self.processed_data = df
        
        print("Data processing complete!")
        print(f"Final processed data shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        return df

# Test the processor
if __name__ == "__main__":
    processor = SWISDataProcessor()
    processed_df = processor.process_swis_data('data/raw_data/sample_swis_data.csv')
    
    if processed_df is not None:
        # Save processed data
        processed_df.to_csv('data/processed_data/processed_swis_data.csv')
        print("Processed data saved successfully!")