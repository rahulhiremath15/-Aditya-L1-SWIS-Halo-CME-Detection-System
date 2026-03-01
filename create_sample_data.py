import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_swis_data():
    """Create synthetic SWIS data for development"""
    # Create time series from Aug 1, 2024 for 30 days
    start_date = datetime(2024, 8, 1)
    end_date = start_date + timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
    n_points = len(timestamps)
    
    # Create synthetic solar wind parameters
    np.random.seed(42)  # For reproducible results
    data = {
        'timestamp': timestamps,
        'proton_flux': np.random.normal(2e8, 5e7, n_points),  # particles/cm²/s
        'alpha_flux': np.random.normal(8e6, 2e6, n_points),  # particles/cm²/s
        'proton_density': np.random.normal(7, 3, n_points),  # particles/cm³
        'proton_velocity': np.random.normal(400, 100, n_points),  # km/s
        'proton_temperature': np.random.normal(5e4, 2e4, n_points),  # K
        'alpha_temperature': np.random.normal(1e5, 3e4, n_points),  # K
    }
    df = pd.DataFrame(data)
    
    # Add some CME events (synthetic)
    cme_events = [
        (datetime(2024, 8, 5, 12, 0), datetime(2024, 8, 5, 15, 0)),
        (datetime(2024, 8, 12, 8, 30), datetime(2024, 8, 12, 11, 45)),
        (datetime(2024, 8, 20, 14, 15), datetime(2024, 8, 20, 17, 30)),
    ]
    
    for start_time, end_time in cme_events:
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        # Enhance parameters during CME events
        df.loc[mask, 'alpha_flux'] *= np.random.uniform(3, 8, mask.sum())
        df.loc[mask, 'proton_velocity'] += np.random.uniform(100, 300, mask.sum())
        df.loc[mask, 'proton_temperature'] *= np.random.uniform(1.5, 3, mask.sum())
    
    return df

# Create and save sample data
sample_data = create_sample_swis_data()
sample_data.to_csv('data/raw_data/sample_swis_data.csv', index=False)
print("Sample SWIS data created successfully!")