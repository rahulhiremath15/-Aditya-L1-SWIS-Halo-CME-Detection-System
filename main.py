#!/usr/bin/env python3
"""
Bharatiya Antariksh Hackathon 2025
CME Detection System - Main Entry Point
"""

from src.visualizer import CMEVisualizer
from src.cme_detector import CMEEventDetector
from src.ml_models import CMEModelTrainer
from src.data_processor import SWISDataProcessor
import sys
import os
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main function to run the CME detection pipeline"""

    parser = argparse.ArgumentParser(
        description="Aditya-L1 SWIS Halo CME Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --sample                    # Run with sample data
  python main.py --file data.csv --visualize # Process custom data with plots
  python main.py --train                     # Train new models
  python main.py --web                       # Launch web interface
        """
    )

    parser.add_argument('--file', type=str,
                        help="Path to SWIS data file (CSV format)")
    parser.add_argument('--sample', action='store_true',
                        help="Run with built-in sample data")
    parser.add_argument('--train', action='store_true',
                        help="Train new ML models")
    parser.add_argument('--visualize', action='store_true',
                        help="Generate visualization plots")
    parser.add_argument('--web', action='store_true',
                        help="Launch web dashboard")
    parser.add_argument('--output', type=str,
                        default='results', help="Output directory")

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print(" BHARATIYA ANTARIKSH HACKATHON 2025 ")
    print("   Aditya-L1 SWIS Halo CME Detection System")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Step 8 Create Main Integration Script
        if args.train:
            print(" TRAINING MODE: Building ML models...")
            trainer = CMEModelTrainer()
            trainer.train_all_models(
                'data/processed_data/processed_swis_data.csv')
            print("✅ Model training completed!")
            return

        if args.web:
            print(" WEB MODE: Launching dashboard...")
            launch_web_dashboard()
            return

        # Determine input file
        if args.sample:
            input_file = 'data/raw_data/sample_swis_data.csv'
            print(f" Using sample data: {input_file}")
        elif args.file:
            input_file = args.file
            print(f" Using custom data: {input_file}")
        else:
            print("❌ Error: Please specify --file or --sample")
            parser.print_help()
            return

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file not found: {input_file}")
            if args.sample:
                print(
                    " Tip: Run 'python create_sample_data.py' first to create sample data.")
            return

        # Initialize modules
        print("\n Initializing system modules...")
        processor = SWISDataProcessor()
        detector = CMEEventDetector()
        visualizer = CMEVisualizer(args.output)

        # Step 1: Process data
        print("\n Step 1: Processing SWIS data...")
        processed_df = processor.process_swis_data(input_file)

        if processed_df is None:
            print("❌ Error: Data processing failed")
            return

        # Save processed data
        processed_file = os.path.join(
            'data', 'processed_data', 'processed_swis_data.csv')
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        processed_df.to_csv(processed_file)
        print(f" Processed data saved to: {processed_file}")

        # Step 2: Detect CME events
        print("\n Step 2: Detecting CME events...")
        detection_results = detector.detect_cme_events(processed_df)

        # Step 3: Generate reports and visualizations
        print(f"\n Step 3: Generating results...")

        # Print summary
        summary = detection_results['summary']
        print(f"   Total CME events detected: {summary['total_events']}")
        print(
            f"   Total detection points: {summary['total_detection_points']}")
        print(f"   Detection rate: {summary['detection_rate']:.4f}")

        # Save results
        results_file = os.path.join(args.output, 'detection_results.json')
        os.makedirs(args.output, exist_ok=True)

        # Create visualizations if requested
        if args.visualize:
            print("\n Creating visualizations...")
            visualizer.create_all_visualizations(
                processed_df, detection_results)
        else:
            # Always save the report
            visualizer.save_event_report(detection_results)

        # Print event details
        if detection_results['events']:
            print(f"\n DETECTED EVENTS:")
            print("-" * 50)
            for i, event in enumerate(detection_results['events']):
                print(f"Event {i+1}:")
                print(f"   Time: {event['start_time']} to {event['end_time']}")
                print(f"     Duration: {event['duration_hours']} hours")
                print(f"   Max Velocity: {event['max_velocity']:.0f} km/s")
                print(f"   Max Temperature: {event['max_temperature']:.0f} K")
                print(
                    f"   Max α/p Ratio: {event['max_alpha_proton_ratio']:.3f}")
                print(f"   Confidence: {event['confidence']}")
                print(f"   Type: {event['event_type']}")
                print()

        print(f"✅ Analysis complete! Results saved to: {args.output}/")
        print(f" Check the following files:")
        print(f"   - {args.output}/reports/cme_detection_report.json")
        print(f"   - {args.output}/reports/event_summary.txt")
        if args.visualize:
            print(f"   - {args.output}/plots/ (visualization plots)")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def launch_web_dashboard():
    from cme_web_app import app
    app.run(debug=True)
    print("Launching Flask web dashboard...")
    print("Open your browser to: http://localhost:5000")


if __name__ == "__main__":
    main()
