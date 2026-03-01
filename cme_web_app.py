# cme_web_app.py

from flask import Flask, render_template, send_from_directory, jsonify
import os
import json
import pandas as pd

app = Flask(__name__)
REPORT_PATH = "results/reports/cme_detection_report.json"
PLOTS_DIR = "results/plots"

@app.route("/")
def dashboard():
    # Load CME detection report
    if not os.path.exists(REPORT_PATH):
        return render_template("dashboard.html", 
                           summary={"total_events": 0, "total_detection_points": 0, "detection_rate": 0.0},
                           events=[],
                           thresholds={},
                           baseline_stats={},
                           plots_available=False)

    try:
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading report: {e}")
        return render_template("dashboard.html", 
                           summary={"total_events": 0, "total_detection_points": 0, "detection_rate": 0.0},
                           events=[],
                           thresholds={},
                           baseline_stats={},
                           plots_available=False)

    # Extract summary stats
    summary = report.get("summary", {})
    events = report.get("detected_events", [])
    thresholds = report.get("detection_thresholds", {})
    baseline_stats = report.get("baseline_statistics", {})
    
    # Check if plots exist
    plots_available = os.path.exists(PLOTS_DIR) and len(os.listdir(PLOTS_DIR)) > 0

    return render_template("dashboard.html",
                           summary=summary,
                           events=events,
                           thresholds=thresholds,
                           baseline_stats=baseline_stats,
                           plots_available=plots_available)

@app.route("/plots/<filename>")
def get_plot(filename):
    try:
        return send_from_directory(PLOTS_DIR, filename)
    except FileNotFoundError:
        return "Plot not found", 404

@app.route("/api/status")
def api_status():
    """API endpoint for system status"""
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r") as f:
                report = json.load(f)
            return jsonify({
                "status": "ready",
                "report_loaded": True,
                "events_count": len(report.get("detected_events", []))
            })
        except:
            return jsonify({"status": "error", "report_loaded": False})
    else:
        return jsonify({"status": "no_report", "report_loaded": False})

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
