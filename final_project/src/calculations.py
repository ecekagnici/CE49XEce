# src/calculations.py

import json
import pandas as pd


class LCACalculator:
    """
    Life Cycle Assessment calculator for computing environmental impacts.
    
    This class handles the core computational logic of the LCA tool, taking product
    data and impact factors to calculate comprehensive environmental impacts across
    different life cycle stages. It provides functionality for individual impact
    calculations, aggregation, normalization, and comparative analysis.
    """

    def __init__(self, impact_factors_path):
        """
        Initialize the calculator with impact factors from a JSON file.
        
        The impact factors file should contain nested dictionaries with the structure:
        {material_type: {life_cycle_stage: {impact_metric: factor_value}}}
        
        Args:
            impact_factors_path (str): Path to JSON file containing impact factors
        """
        with open(impact_factors_path, 'r') as f:
            self.impact_factors = json.load(f)

    def calculate_impacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate environmental impacts for each product and life cycle stage.
        
        This method multiplies the quantity of each product by the appropriate impact
        factors based on material type and life cycle stage. It creates three new
        columns for carbon, energy, and water impacts.
        
        The calculation follows this formula:
        impact = quantity_kg * impact_factor[material][stage][metric]
        
        Args:
            df (pd.DataFrame): Product data with required columns including
                              material_type, life_cycle_stage, and quantity_kg
        
        Returns:
            pd.DataFrame: Original data with added impact columns (carbon_impact,
                         energy_impact, water_impact)
        """
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Create temporary columns with normalized material and stage names
        # This ensures consistent matching with impact factor keys
        df['_mat'] = df['material_type'].str.lower()
        df['_stage'] = df['life_cycle_stage'].str.lower()

        # Calculate each type of environmental impact
        for metric in ('carbon_impact', 'energy_impact', 'water_impact'):
            # For each row, multiply quantity by the appropriate impact factor
            # Use .get() with default empty dict to handle missing materials/stages gracefully
            df[metric] = df.apply(
                lambda r: r['quantity_kg']
                          * self.impact_factors
                              .get(r['_mat'], {})
                              .get(r['_stage'], {})
                              .get(metric, 0),
                axis=1
            )

        # Clean up temporary columns used for matching
        # Use errors='ignore' to prevent issues if columns don't exist
        df.drop(columns=['_mat', '_stage'], inplace=True, errors='ignore')
        return df

    def calculate_total_impacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate environmental impacts by product across all life cycle stages.
        
        This method groups the detailed impact data by product and sums up all
        impacts to provide a total environmental footprint for each product.
        This is useful for comparing the overall environmental performance
        of different products.
        
        Args:
            df (pd.DataFrame): Product data with calculated impacts
        
        Returns:
            pd.DataFrame: Aggregated impacts grouped by product_id and product_name
        """
        # Group by product identifier and name, then sum all impact metrics
        total = df.groupby(['product_id', 'product_name'], as_index=False).agg({
            'carbon_impact': 'sum',
            'energy_impact': 'sum', 
            'water_impact': 'sum',
            'waste_generated_kg': 'sum'
        })
        return total

    def normalize_impacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize impact values to a 0-1 scale using min-max normalization.
        
        This method scales all impact values to a common range, making it easier
        to compare different types of impacts (carbon, energy, water) that may
        have very different absolute values. Each impact type is normalized
        independently using the formula: (value - min) / (max - min)
        
        Args:
            df (pd.DataFrame): Data with impact columns to normalize
        
        Returns:
            pd.DataFrame: Data with normalized impact values between 0 and 1
        """
        # Create a copy to preserve original data
        norm = df.copy()
        
        # Normalize each impact metric independently
        for metric in ('carbon_impact', 'energy_impact', 'water_impact'):
            # Find the minimum and maximum values for this metric
            mn = norm[metric].min()
            mx = norm[metric].max()
            
            # Only normalize if there's actual variation in the data
            if mx > mn:
                # Apply min-max normalization formula
                norm[metric] = (norm[metric] - mn) / (mx - mn)
            else:
                # If all values are the same, set to 0
                norm[metric] = 0
        return norm

    def compare_alternatives(self, df: pd.DataFrame, product_ids):
        """
        Compare environmental impacts between alternative products.
        
        This method calculates relative impacts by comparing each product to a
        baseline (the first product in the list). It adds "_relative" columns
        showing how each product's impact compares to the baseline as a ratio.
        A value of 1.0 means equal to baseline, >1.0 means worse, <1.0 means better.
        
        Args:
            df (pd.DataFrame): Product data with calculated impacts
            product_ids (list): List of product IDs to compare, first one serves as baseline
        
        Returns:
            pd.DataFrame: Comparison data with both absolute and relative impact values
        """
        # First get total impacts for all products
        total = self.calculate_total_impacts(df)
        
        # Filter to only the products we want to compare
        comp = total[total['product_id'].isin(product_ids)].copy()
        
        # Ensure products are in the same order as requested
        comp = comp.set_index('product_id').loc[product_ids].reset_index()

        # Get the baseline product (first in the list) for comparison
        baseline = comp.loc[comp['product_id'] == product_ids[0], :]
        
        # Calculate relative impacts compared to baseline
        for metric in ('carbon_impact', 'energy_impact', 'water_impact'):
            # Extract baseline value for this metric
            base_val = float(baseline[metric].iloc[0])
            
            # Calculate ratio to baseline, handling division by zero
            if base_val != 0:
                comp[f'{metric}_relative'] = comp[metric] / base_val
            else:
                comp[f'{metric}_relative'] = None

        return comp