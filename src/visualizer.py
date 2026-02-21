import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CMEVisualizer:
    """Visualize CME detection results"""

    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)

    def plot_full_timeseries(self, df, results):
        """Plot complete time series with detected events"""
        print("Creating full time series plot...")

        fig, axes = plt.subplots(4, 1, figsize=(20, 16))

        # Plot 1: Alpha-proton ratio
        axes[0].plot(df.index, df['alpha_proton_ratio'], 'b-', alpha=0.7, linewidth=1)
        axes[0].set_ylabel('Alpha/Proton Ratio')
        axes[0].set_title('SWIS Solar Wind Parameters with CME Events', fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Proton velocity
        axes[1].plot(df.index, df['proton_velocity'], 'g-', alpha=0.7, linewidth=1)
        axes[1].set_ylabel('Proton Velocity\n(km/s)')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Proton temperature
        axes[2].plot(df.index, df['proton_temperature']/1000, 'r-', alpha=0.7, linewidth=1)
        axes[2].set_ylabel('Proton Temperature\n(1000 K)')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Detection scores
        axes[3].plot(df.index, df['ensemble_score'], color='purple', alpha=0.8, linewidth=1.5)
        axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Detection Threshold')
        axes[3].set_ylabel('Ensemble Score')
        axes[3].set_xlabel('Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Highlight detected events
        for event in results['events']:
            start_time = pd.to_datetime(event['start_time'])
            end_time = pd.to_datetime(event['end_time'])
            for ax in axes:
                ax.axvspan(start_time, end_time, alpha=0.2, color='red', label='CME Event')

        # Add legend to first plot (avoid duplicate labels)
        if results['events']:
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'plots', 'full_timeseries_with_events.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Full time series plot saved to {plot_path}")

    def plot_event_details(self, df, results):
        """Create detailed plots for each detected event"""
        print("Creating detailed event plots...")

        for i, event in enumerate(results['events']):
            start_time = pd.to_datetime(event['start_time'])
            end_time = pd.to_datetime(event['end_time'])

            # Extend time window for context (2 hours before and after)
            context_start = start_time - pd.Timedelta(hours=2)
            context_end = end_time + pd.Timedelta(hours=2)

            event_data = df[context_start:context_end]

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle(f'CME Event {i+1}: {event["start_time"]} to {event["end_time"]}',
                         fontsize=16, fontweight='bold')

            # Plot 1: Alpha-proton ratio
            axes[0,0].plot(event_data.index, event_data['alpha_proton_ratio'], 'b-', linewidth=1)
            axes[0,0].axvspan(start_time, end_time, alpha=0.2, color='red', label='CME Period')
            axes[0,0].set_ylabel('Alpha/Proton Ratio')
            axes[0,0].set_title('Alpha-Proton Ratio Enhancement')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Plot 2: Proton velocity
            axes[0,1].plot(event_data.index, event_data['proton_velocity'], 'g-', linewidth=1)
            axes[0,1].axvspan(start_time, end_time, alpha=0.2, color='red')
            axes[0,1].set_ylabel('Velocity (km/s)')
            axes[0,1].set_title('Proton Velocity Profile')
            axes[0,1].grid(True, alpha=0.3)

            # Plot 3: Temperature
            axes[1,0].plot(event_data.index, event_data['proton_temperature']/1000, 'r-', linewidth=1)
            axes[1,0].axvspan(start_time, end_time, alpha=0.2, color='red')
            axes[1,0].set_ylabel('Temperature (1000 K)')
            axes[1,0].set_title('Proton Temperature')
            axes[1,0].grid(True, alpha=0.3)

            # Plot 4: Detection scores
            axes[1,1].plot(event_data.index, event_data['ensemble_score'], color='purple', linewidth=1.5)
            axes[1,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
            axes[1,1].axvspan(start_time, end_time, alpha=0.2, color='red')
            axes[1,1].set_ylabel('Ensemble Score')
            axes[1,1].set_title('Detection Confidence')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            # Plot 5: Individual model predictions
            axes[2,0].plot(event_data.index, event_data['rf_probability'], label='RF Probability')
            axes[2,0].plot(event_data.index, event_data['svm_probability'], label='SVM Probability')
            axes[2,0].plot(event_data.index, event_data['stat_detection'], label='Statistical Detection')
            axes[2,0].axvspan(start_time, end_time, alpha=0.2, color='red')
            axes[2,0].set_ylabel('Probability/Detection')
            axes[2,0].set_title('Individual Model Results')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)

            # Plot 6: Event summary text
            axes[2,1].axis('off')
            summary_text = f"""
Event Summary:
Duration: {event['duration_hours']} hours
Max α/p ratio: {event['max_alpha_proton_ratio']:.3f}
Max velocity: {event['max_velocity']:.0f} km/s
Max temperature: {event['max_temperature']:.0f} K
Confidence: {event['confidence']}
Type: {event['event_type']}
Mean score: {event['mean_ensemble_score']:.3f}
            """
            axes[2,1].text(0.1, 0.5, summary_text, fontsize=12,
                           verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'plots', f'event_{i+1}_detail.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Event {i+1} detail plot saved to {plot_path}")

    def plot_statistics_summary(self, results):
        """Create statistical summary plots"""
        print("Creating statistics summary...")

        if not results['events']:
            print("No events to plot statistics for.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CME Detection Statistics Summary', fontsize=16, fontweight='bold')

        # Extract event characteristics
        durations = [event['duration_hours'] for event in results['events']]
        max_ratios = [event['max_alpha_proton_ratio'] for event in results['events']]
        max_velocities = [event['max_velocity'] for event in results['events']]
        max_temps = [event['max_temperature']/1000 for event in results['events']]

        # Plot 1: Event durations
        axes[0,0].hist(durations, bins=5, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_xlabel('Duration (hours)')
        axes[0,0].set_ylabel('Number of Events')
        axes[0,0].set_title('Event Duration Distribution')
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Maximum alpha-proton ratios
        axes[0,1].hist(max_ratios, bins=5, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_xlabel('Max Alpha/Proton Ratio')
        axes[0,1].set_ylabel('Number of Events')
        axes[0,1].set_title('Alpha-Proton Ratio Distribution')
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Maximum velocities
        axes[1,0].hist(max_velocities, bins=5, alpha=0.7, color='red', edgecolor='black')
        axes[1,0].set_xlabel('Max Velocity (km/s)')
        axes[1,0].set_ylabel('Number of Events')
        axes[1,0].set_title('Velocity Distribution')
        axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Maximum temperatures
        axes[1,1].hist(max_temps, bins=5, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_xlabel('Max Temperature (1000 K)')
        axes[1,1].set_ylabel('Number of Events')
        axes[1,1].set_title('Temperature Distribution')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'plots', 'statistics_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Statistics summary plot saved to {plot_path}")

    def save_event_report(self, results):
        """Save detailed event report as JSON"""
        print("Saving event report...")

        # Create comprehensive report
        report = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': results['summary'],
            'baseline_statistics': results['baseline_stats'],
            'detection_thresholds': results['thresholds'],
            'detected_events': results['events']
        }

        report_path = os.path.join(self.output_dir, 'reports', 'cme_detection_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, default=str)

        print(f"Event report saved to {report_path}")

        # Also create a human-readable summary
        summary_path = os.path.join(self.output_dir, 'reports', 'event_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("CME DETECTION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis completed: {report['generation_time']}\n")
            f.write(f"Total events detected: {results['summary']['total_events']}\n")
            f.write(f"Total detection points: {results['summary']['total_detection_points']}\n")
            f.write(f"Detection rate: {results['summary']['detection_rate']:.4f}\n\n")

            if results['events']:
                f.write("DETECTED EVENTS:\n")
                f.write("-" * 20 + "\n")

                for i, event in enumerate(results['events']):
                    f.write(f"\nEvent {i+1}:\n")
                    f.write(f"  Start: {event['start_time']}\n")
                    f.write(f"  End: {event['end_time']}\n")
                    f.write(f"  Duration: {event['duration_hours']} hours\n")
                    f.write(f"  Type: {event['event_type']}\n")
                    f.write(f"  Confidence: {event['confidence']}\n")
                    f.write(f"  Max velocity: {event['max_velocity']:.0f} km/s\n")
                    f.write(f"  Max α/p ratio: {event['max_alpha_proton_ratio']:.3f}\n")
            else:
                f.write("No CME events detected in this dataset.\n")

        print(f"Human-readable summary saved to {summary_path}")

    def create_all_visualizations(self, df, results):
        """Create all visualization outputs"""
        print("Creating all visualizations...")

        self.plot_full_timeseries(df, results)
        self.plot_event_details(df, results)
        self.plot_statistics_summary(results)
        self.save_event_report(results)

        print("All visualizations complete!")

# Test the visualizer
if __name__ == "__main__":
    # This would be called by main.py
    print("Visualizer module ready!")
