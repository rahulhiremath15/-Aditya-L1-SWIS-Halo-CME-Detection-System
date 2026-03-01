import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os


class CMEVisualizationDashboard:

    def __init__(self):
        self.colors = {
            'cme_detected': '#ff6b6b',
            'alpha_ratio': '#45b7d1',
            'velocity': '#96ceb4',
            'temperature': '#ffeaa7',
            'confidence': '#fd79a8'
        }

    def create_comprehensive_dashboard(self, data, results):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Alpha-Proton Ratio",
                "Solar Wind Velocity",
                "Detected CME Periods"
            ],
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Plot alpha-proton ratio
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['alpha_proton_ratio'],
                mode='lines',
                name='Alpha/Proton Ratio'
            ),
            row=1, col=1
        )

        # Plot velocity
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['proton_velocity'],
                mode='lines',
                name='Velocity'
            ),
            row=2, col=1
        )

        # Highlight detected events
        events = results.get('events', [])

        for event in events:
            start = pd.to_datetime(event['start_time'])
            end = pd.to_datetime(event['end_time'])

            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="red",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=1, col=1
            )

            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="red",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2, col=1
            )

        fig.update_layout(
            height=900,
            title="Aditya-L1 SWIS CME Detection Dashboard"
        )

        return fig

    def save_dashboard_html(self, fig, filename="cme_dashboard.html"):

        output_dir = os.path.join("static", "plots")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)

        print(f"Dashboard saved to: {output_path}")
        return output_path