#!/usr/bin/env python3
"""
Life Cycle Assessment (LCA) Analysis Pipeline

This script provides a comprehensive command-line interface for performing
complete LCA analysis. It orchestrates the entire workflow from data input
through validation, impact calculations, and visualization output generation.

The pipeline is designed to be robust, user-friendly, and production-ready,
with proper error handling and informative feedback throughout the process.
"""

import argparse
import sys
from pathlib import Path
import os

# Configure Python path to enable imports from the src directory
# This allows the script to find our custom modules regardless of where it's run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
# Import our custom LCA analysis modules
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer


def main():
    """
    Main execution function for the LCA analysis pipeline.
    
    This function coordinates the entire LCA workflow:
    1. Parse command line arguments
    2. Set up output directory structure
    3. Load and validate input data
    4. Perform impact calculations
    5. Generate comprehensive visualizations
    6. Export results to CSV files
    
    The process is designed to be fault-tolerant, providing clear error messages
    and gracefully handling various failure scenarios.
    """
    # Set up command line argument parsing with comprehensive help
    parser = argparse.ArgumentParser(
        description="Complete LCA pipeline: reads product data, validates it, calculates environmental impacts, and generates both CSV reports and visualization plots in the specified output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -d data/raw/products.csv -f data/raw/impact_factors.json
  python main.py --data mydata.xlsx --factors impacts.json --output my_analysis
        """
    )
    
    # Define required input data file argument
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to your product data file. Supports CSV, Excel (.xls/.xlsx), or JSON formats. Must contain all required LCA columns including product info, lifecycle stages, and environmental data."
    )
    
    # Define required impact factors file argument
    parser.add_argument(
        "--factors", "-f", required=True,
        help="Path to impact_factors.json file containing environmental impact coefficients for different materials and lifecycle stages."
    )
    
    # Define optional output directory argument
    parser.add_argument(
        "--output", "-o", default="results",
        help="Name of the output folder where results will be saved. This folder will be created under the 'data/' directory if it doesn't exist. Default: 'results'"
    )
    
    # Parse the provided command line arguments
    args = parser.parse_args()

    # Determine the project's data directory structure
    # We expect input data to be in data/raw/, so we can infer the project root
    data_path = Path(args.data)
    project_data_dir = data_path.parent.parent  # data/raw/file.csv -> data/
    
    # Create output directory within the project data structure
    outdir = project_data_dir / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting LCA analysis pipeline...")
    print(f"📂 Input data: {args.data}")
    print(f"🔬 Impact factors: {args.factors}")
    print(f"📊 Output directory: {outdir.resolve()}")

    # Phase 1: Data Loading and Validation
    print(f"\n📖 Phase 1: Loading and validating input data...")
    
    # Initialize the data input handler
    di = DataInput()
    
    # Attempt to load the product data file
    try:
        df = di.read_data(args.data)
        print(f"✅ Successfully loaded {len(df)} rows of product data")
    except Exception as e:
        print(f"❌ Failed to read data file: {e}", file=sys.stderr)
        print("💡 Please check that the file exists and is in a supported format (CSV, Excel, or JSON)", file=sys.stderr)
        sys.exit(1)

    # Validate the loaded data for completeness and correctness
    if not di.validate_data(df):
        print("❌ Data validation failed — please check required columns/types/rates.", file=sys.stderr)
        print("💡 Ensure all required columns are present and numeric columns contain valid numbers", file=sys.stderr)
        print("💡 Check that end-of-life rates (recycling, landfill, incineration) sum to ≤ 1.0", file=sys.stderr)
        sys.exit(1)
    
    print("✅ Data validation passed - all required columns and data types are correct")

    # Phase 2: Impact Calculations
    print(f"\n🧮 Phase 2: Calculating environmental impacts...")
    
    # Initialize the LCA calculator with impact factors
    lc = LCACalculator(args.factors)
    
    # Calculate detailed impacts for each product and lifecycle stage
    df_imp = lc.calculate_impacts(df)
    df_imp.to_csv(outdir / "detailed_impacts.csv", index=False)
    print("✅ Detailed impacts calculated and saved")

    # Calculate total impacts aggregated by product
    df_tot = lc.calculate_total_impacts(df_imp)
    df_tot.to_csv(outdir / "total_impacts.csv", index=False)
    print("✅ Total impacts by product calculated and saved")

    # Generate normalized impacts for easier comparison
    df_norm = lc.normalize_impacts(df_tot)
    df_norm.to_csv(outdir / "normalized_impacts.csv", index=False)
    print("✅ Normalized impacts calculated and saved")

    # Perform product comparison analysis if multiple products exist
    unique_ids = df_tot["product_id"].unique().tolist()
    comp_ids = []
    if len(unique_ids) >= 2:
        comp_ids = unique_ids[:2]  # Compare first two products
        df_comp = lc.compare_alternatives(df_imp, comp_ids)
        df_comp.to_csv(outdir / "comparison.csv", index=False)
        print(f"✅ Product comparison analysis completed for {comp_ids[0]} vs {comp_ids[1]}")
    else:
        print("ℹ️  Only one product found - skipping comparative analysis")

    # Phase 3: Visualization Generation
    print(f"\n📊 Phase 3: Generating visualizations...")
    
    # Initialize the visualization engine
    viz = LCAVisualizer()
    
    # Generate impact breakdown by material type
    try:
        fig = viz.plot_impact_breakdown(df_imp, "carbon_impact", "material_type")
        fig.savefig(outdir / "breakdown_by_material_type.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✅ Material impact breakdown chart created")
    except Exception as e:
        print(f"⚠️  Could not create material breakdown chart: {e}", file=sys.stderr)

    # Generate lifecycle impact charts for the first product
    if unique_ids:
        pid = unique_ids[0]
        try:
            fig = viz.plot_life_cycle_impacts(df_imp, pid)
            fig.savefig(outdir / f"lifecycle_{pid}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Lifecycle impact analysis chart created for {pid}")
        except Exception as e:
            print(f"⚠️  Could not create lifecycle chart for {pid}: {e}", file=sys.stderr)

    # Generate product comparison visualization
    if comp_ids:
        try:
            fig = viz.plot_product_comparison(df_imp, comp_ids)
            fig.savefig(outdir / "product_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("✅ Product comparison chart created")
        except Exception as e:
            print(f"⚠️  Could not create product comparison chart: {e}", file=sys.stderr)

    # Generate end-of-life management pie chart
    if unique_ids:
        pid = unique_ids[0]
        try:
            fig = viz.plot_end_of_life_breakdown(df_imp, pid)
            fig.savefig(outdir / f"end_of_life_{pid}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ End-of-life breakdown chart created for {pid}")
        except Exception as e:
            print(f"⚠️  Could not create end-of-life chart for {pid}: {e}", file=sys.stderr)

    # Generate impact correlation heatmap
    try:
        fig = viz.plot_impact_correlation(df_imp)
        fig.savefig(outdir / "impact_correlation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✅ Impact correlation heatmap created")
    except Exception as e:
        print(f"⚠️  Could not create correlation heatmap: {e}", file=sys.stderr)

    # Analysis Complete
    print(f"\n🎉 LCA analysis pipeline completed successfully!")
    print(f"📁 All results have been saved to: {outdir.resolve()}")
    print(f"📄 CSV files: detailed_impacts.csv, total_impacts.csv, normalized_impacts.csv")
    if comp_ids:
        print(f"📄 Comparison file: comparison.csv")
    print(f"🖼️  Visualization files: Multiple PNG charts for comprehensive analysis")


if __name__ == "__main__":
    """
    Entry point for command-line execution.
    
    This ensures the main function only runs when the script is executed directly,
    not when it's imported as a module. This is a Python best practice for
    creating reusable scripts.
    """
    main()