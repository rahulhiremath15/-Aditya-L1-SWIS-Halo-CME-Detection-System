# Create the web interface and visualization system

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

class CMEVisualizationDashboard:
    """
    Interactive visualization dashboard for CME detection results
    Features real-time plotting, detection overlays, and statistical analysis
    """
    
    def __init__(self):
        self.colors = {
            'cme_detected': '#ff6b6b',
            'normal_wind': '#4ecdc4', 
            'alpha_ratio': '#45b7d1',
            'velocity': '#96ceb4',
            'temperature': '#ffeaa7',
            'confidence': '#fd79a8'
        }
    
    def create_comprehensive_dashboard(self, data, results):
        """Create comprehensive CME detection dashboard"""
        print("Creating interactive CME detection dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Alpha-Proton Ratio vs Time', 'CME Detection Timeline',
                'Solar Wind Velocity', 'Proton Temperature', 
                'Detection Confidence Score', 'Event Characteristics',
                'Performance Metrics', 'Statistical Summary'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Add traces for each subplot
        self._add_alpha_ratio_plot(fig, data, results, row=1, col=1)
        self._add_detection_timeline(fig, data, results, row=1, col=2)
        self._add_velocity_plot(fig, data, results, row=2, col=1)
        self._add_temperature_plot(fig, data, results, row=2, col=2)
        self._add_confidence_plot(fig, data, results, row=3, col=1)
        self._add_event_characteristics(fig, results, row=3, col=2)
        self._add_performance_table(fig, results, row=4, col=1)
        self._add_statistics_table(fig, data, results, row=4, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title={
                'text': 'Aditya-L1 SWIS CME Detection Dashboard',
                'x': 0.5,
                'font': {'size': 20}
            },
            showlegend=True
        )
        
        return fig
    
    def _add_alpha_ratio_plot(self, fig, data, results, row, col):
        """Add alpha-proton ratio time series with CME highlights"""
        
        # Main alpha-proton ratio
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['alpha_proton_ratio'],
                mode='lines',
                name='Alpha/Proton Ratio',
                line=dict(color=self.colors['alpha_ratio'], width=1),
                opacity=0.7
            ),
            row=row, col=col
        )
        
        # CME detection overlay
        cme_mask = results['ensemble_result']['cme_detected']
        if cme_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'][cme_mask],
                    y=data['alpha_proton_ratio'][cme_mask],
                    mode='markers',
                    name='CME Detected',
                    marker=dict(
                        color=self.colors['cme_detected'],
                        size=3,
                        symbol='circle'
                    )
                ),
                row=row, col=col
            )
        
        # Enhancement threshold line
        threshold = detection_engine.thresholds.get('alpha_proton_ratio', 0.08)
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"CME Threshold ({threshold:.3f})",
            row=row, col=col
        )
    
    def _add_detection_timeline(self, fig, data, results, row, col):
        """Add CME detection timeline"""
        
        # Create detection state plot
        detection_state = results['ensemble_result']['cme_detected'].astype(int)
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=detection_state,
                mode='lines',
                name='CME Detection State',
                line=dict(color=self.colors['cme_detected'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(255, 107, 107, 0.3)"
            ),
            row=row, col=col
        )
        
        # Add event markers
        events = results['cme_events']
        if events:
            event_times = [pd.to_datetime(event['start_time']) for event in events]
            event_labels = [f"Event {event['event_id']}" for event in events]
            
            fig.add_trace(
                go.Scatter(
                    x=event_times,
                    y=[1.1] * len(event_times),
                    mode='markers+text',
                    name='CME Events',
                    text=event_labels,
                    textposition='top center',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-up'
                    )
                ),
                row=row, col=col
            )
    
    def _add_velocity_plot(self, fig, data, results, row, col):
        """Add solar wind velocity plot"""
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['proton_velocity'],
                mode='lines',
                name='Proton Velocity',
                line=dict(color=self.colors['velocity'], width=1)
            ),
            row=row, col=col
        )
        
        # CME velocity overlay
        cme_mask = results['ensemble_result']['cme_detected']
        if cme_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'][cme_mask],
                    y=data['proton_velocity'][cme_mask],
                    mode='markers',
                    name='CME Velocity',
                    marker=dict(
                        color=self.colors['cme_detected'],
                        size=2
                    )
                ),
                row=row, col=col
            )
    
    def _add_temperature_plot(self, fig, data, results, row, col):
        """Add proton temperature plot"""
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['proton_temperature'],
                mode='lines',
                name='Proton Temperature',
                line=dict(color=self.colors['temperature'], width=1)
            ),
            row=row, col=col
        )
        
        # CME temperature overlay
        cme_mask = results['ensemble_result']['cme_detected']
        if cme_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'][cme_mask],
                    y=data['proton_temperature'][cme_mask],
                    mode='markers',
                    name='CME Temperature',
                    marker=dict(
                        color=self.colors['cme_detected'],
                        size=2
                    )
                ),
                row=row, col=col
            )
    
    def _add_confidence_plot(self, fig, data, results, row, col):
        """Add detection confidence score plot"""
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=results['ensemble_result']['confidence_score'],
                mode='lines',
                name='Confidence Score',
                line=dict(color=self.colors['confidence'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(253, 121, 168, 0.3)"
            ),
            row=row, col=col
        )
        
        # Detection threshold line
        threshold = results['ensemble_result']['threshold_used']
        fig.add_hline(
            y=threshold,
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Detection Threshold ({threshold})",
            row=row, col=col
        )
    
    def _add_event_characteristics(self, fig, results, row, col):
        """Add event characteristics bar chart"""
        
        events = results['cme_events']
        if not events:
            return
            
        event_types = {}
        confidence_levels = {}
        
        for event in events:
            event_type = event.get('event_type', 'Unknown')
            confidence = event.get('confidence_level', 'Unknown')
            
            event_types[event_type] = event_types.get(event_type, 0) + 1
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
        
        # Event types
        fig.add_trace(
            go.Bar(
                x=list(event_types.keys()),
                y=list(event_types.values()),
                name='Event Types',
                marker_color='lightblue'
            ),
            row=row, col=col
        )
    
    def _add_performance_table(self, fig, results, row, col):
        """Add performance metrics table"""
        
        performance = results['performance']
        if 'metrics' in performance:
            metrics = performance['metrics']
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                              fill_color='paleturquoise',
                              align='left'),
                    cells=dict(values=[
                        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
                        [f"{metrics['accuracy']:.3f}",
                         f"{metrics['precision']:.3f}",
                         f"{metrics['recall']:.3f}",
                         f"{metrics['f1_score']:.3f}",
                         f"{metrics['specificity']:.3f}"]
                    ],
                    fill_color='lavender',
                    align='left')
                ),
                row=row, col=col
            )
    
    def _add_statistics_table(self, fig, data, results, row, col):
        """Add detection statistics table"""
        
        report = results['report']
        summary = report['detection_summary']
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value'],
                          fill_color='lightgreen',
                          align='left'),
                cells=dict(values=[
                    ['Total Events', 'Detection Rate (%)', 'Avg Duration (hrs)', 
                     'Max Confidence', 'Data Points'],
                    [f"{summary['total_events_detected']}",
                     f"{summary['detection_rate_percent']:.1f}",
                     f"{summary['average_event_duration_hours']:.1f}",
                     f"{summary['max_confidence_score']:.3f}",
                     f"{report['report_metadata']['total_data_points']}"]
                ],
                fill_color='lightgray',
                align='left')
            ),
            row=row, col=col
        )
    
    def create_event_detail_plot(self, data, event, results):
        """Create detailed plot for individual CME event"""
        
        start_idx = event['start_idx']
        end_idx = event['end_idx']
        
        # Extend window for context
        context_hours = 6  # 6 hours before/after
        context_points = context_hours * 60
        
        plot_start = max(0, start_idx - context_points)
        plot_end = min(len(data), end_idx + context_points)
        
        plot_data = data.iloc[plot_start:plot_end].copy()
        plot_results = {
            'cme_detected': results['ensemble_result']['cme_detected'][plot_start:plot_end],
            'confidence_score': results['ensemble_result']['confidence_score'][plot_start:plot_end]
        }
        
        # Create subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"Event {event['event_id']} - Alpha/Proton Ratio",
                "Solar Wind Velocity & Temperature", 
                "Detection Confidence"
            ],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Alpha-proton ratio
        fig.add_trace(
            go.Scatter(
                x=plot_data['timestamp'],
                y=plot_data['alpha_proton_ratio'],
                mode='lines+markers',
                name='Alpha/Proton Ratio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Highlight event period
        event_mask = plot_results['cme_detected']
        fig.add_trace(
            go.Scatter(
                x=plot_data['timestamp'][event_mask],
                y=plot_data['alpha_proton_ratio'][event_mask],
                mode='markers',
                name='CME Period',
                marker=dict(color='red', size=4)
            ),
            row=1, col=1
        )
        
        # Velocity and temperature
        fig.add_trace(
            go.Scatter(
                x=plot_data['timestamp'],
                y=plot_data['proton_velocity'],
                mode='lines',
                name='Velocity (km/s)',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_data['timestamp'],
                y=plot_data['proton_temperature']/1000,  # Scale to thousands
                mode='lines',
                name='Temperature (1000 K)',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # Confidence score
        fig.add_trace(
            go.Scatter(
                x=plot_data['timestamp'],
                y=plot_results['confidence_score'],
                mode='lines',
                name='Confidence Score',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=f"CME Event {event['event_id']} Detailed Analysis",
            showlegend=True
        )
        
        return fig
    
    def save_dashboard_html(self, fig, filename="cme_dashboard.html"):
        """Save dashboard as HTML file"""
        fig.write_html(filename)
        print(f"Dashboard saved as: {filename}")
        return filename

# Create Flask web application for real-time interface
flask_app_code = '''
from flask import Flask, render_template, jsonify, request, send_file
import plotly
import plotly.graph_objects as go
import json
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/realtime_data')
def get_realtime_data():
    """Get real-time SWIS data (simulated)"""
    # In real implementation, this would connect to PRADAN API
    current_time = datetime.now()
    
    # Generate sample real-time data point
    data_point = {
        'timestamp': current_time.isoformat(),
        'alpha_proton_ratio': 0.05 + 0.02 * np.random.random(),
        'proton_velocity': 400 + 100 * np.random.normal(),
        'proton_temperature': 50000 + 20000 * np.random.normal(),
        'cme_probability': np.random.random()
    }
    
    return jsonify(data_point)

@app.route('/api/detect_cme', methods=['POST'])
def detect_cme():
    """Run CME detection on uploaded data"""
    # Implementation would process uploaded SWIS data
    return jsonify({'status': 'success', 'events_detected': 3})

@app.route('/api/cactus_validation')
def cactus_validation():
    """Get CACTUS validation data"""
    # Implementation would fetch CACTUS data
    return jsonify({'validation_score': 0.87, 'matched_events': 5})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
# Test the visualization system
print("="*60)
print("CME VISUALIZATION DASHBOARD")
print("="*60)

# Initialize dashboard
dashboard = CMEVisualizationDashboard()

# Create comprehensive dashboard
dashboard_fig = dashboard.create_comprehensive_dashboard(processed_data, results)

# Save as HTML
html_file = dashboard.save_dashboard_html(dashboard_fig, "aditya_cme_dashboard.html")

# Create detailed plot for first event
if results['cme_events']:
    first_event = results['cme_events'][0]
    event_detail_fig = dashboard.create_event_detail_plot(processed_data, first_event, results)
    event_html_file = dashboard.save_dashboard_html(event_detail_fig, f"cme_event_{first_event['event_id']}_detail.html")
    print(f"Event detail plot saved as: {event_html_file}")

# Save Flask app code for web interface
with open('cme_web_app.py', 'w') as f:
    f.write(flask_app_code)

print(f"Web application code saved as: cme_web_app.py")
print(f"Main dashboard saved as: {html_file}")

print("\nVisualization system created successfully!")
print("To run the web application:")
print("1. Install Flask: pip install flask")
print("2. Run: python cme_web_app.py")
print("3. Open browser to: http://localhost:5000")