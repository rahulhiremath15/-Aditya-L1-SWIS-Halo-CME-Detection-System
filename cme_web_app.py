# cme_web_app.py

from flask import Flask, render_template, send_from_directory
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
        return "Error: Detection report not found. Please run the detection pipeline first."

    with open(REPORT_PATH, "r") as f:
        report = json.load(f)

    # Extract summary stats
    summary = report.get("summary", {})
    events = report.get("detected_events", [])
    thresholds = report.get("detection_thresholds", {})
    baseline_stats = report.get("baseline_statistics", {})

    return render_template("dashboard.html",
                           summary=summary,
                           events=events,
                           thresholds=thresholds,
                           baseline_stats=baseline_stats)

@app.route("/plots/<filename>")
def get_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
