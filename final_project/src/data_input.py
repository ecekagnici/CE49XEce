# src/data_input.py

import json
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt


class DataInput:
    """
    Handles reading and validating both product data and impact factor data.
    
    This class serves as the primary data gateway for the LCA tool, providing
    robust functionality for loading data from multiple file formats and ensuring
    data quality through comprehensive validation. It acts as a protective layer
    between raw input data and the analysis pipeline, catching potential issues
    early in the process.
    """

    # Define all columns that must be present in a valid product dataset
    # These columns represent the complete lifecycle information needed for LCA analysis
    REQUIRED_COLUMNS = [
        'product_id', 'product_name', 'life_cycle_stage', 'material_type',
        'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
        'transport_mode', 'waste_generated_kg',
        'recycling_rate', 'landfill_rate', 'incineration_rate',
        'carbon_footprint_kg_co2e', 'water_usage_liters'
    ]
    
    # Specify which columns must contain numeric data for calculations
    # These columns will be used in mathematical operations and must be quantifiable
    NUMERIC_COLUMNS = [
        'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
        'waste_generated_kg', 'carbon_footprint_kg_co2e', 'water_usage_liters'
    ]
    
    # Define columns representing end-of-life disposal rates
    # These must sum to ≤ 1.0 for each product as they represent percentages
    RATE_COLUMNS = ['recycling_rate', 'landfill_rate', 'incineration_rate']

    def read_data(self, file_path):
        """
        Read product data from various file formats into a pandas DataFrame.
        
        This method provides flexible data input capabilities, automatically detecting
        file format based on extension and applying appropriate parsing methods.
        It supports the most common data formats used in environmental analysis.
        
        Supported formats:
        - CSV: Comma-separated values, most common for tabular data
        - Excel: .xls and .xlsx formats, preserving formatting when needed
        - JSON: Nested data structures, useful for complex hierarchical data
        
        Args:
            file_path (str): Path to the data file to read
            
        Returns:
            pd.DataFrame: Parsed data ready for validation and analysis
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            pandas.errors.ParserError: If file format is corrupted
        """
        # Convert string path to pathlib object for robust path handling
        path = Path(file_path)
        # Get lowercase extension to ensure case-insensitive matching
        ext = path.suffix.lower()

        # Handle CSV files - most common format for environmental data
        if ext == '.csv':
            df = pd.read_csv(path)
        # Handle Excel files - both legacy and modern formats
        elif ext in ('.xls', '.xlsx'):
            df = pd.read_excel(path)
        # Handle JSON files - useful for nested or hierarchical data structures
        elif ext == '.json':
            # Load JSON data and convert to DataFrame
            # Using orient='index' assumes each top-level key represents a row
            with open(path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index')
        else:
            # Reject unsupported file formats with clear error message
            raise ValueError(f"Unsupported file type: {ext}")

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Perform comprehensive validation of product data for LCA analysis.
        
        This method ensures data quality and completeness before proceeding with
        calculations. It checks for structural requirements (required columns),
        data type consistency (numeric where needed), and business logic constraints
        (rate values within valid ranges).
        
        Validation checks performed:
        1. All required columns are present
        2. Numeric columns contain only numeric data
        3. Rate columns are numeric and within [0,1] range
        4. End-of-life rates sum to ≤ 1.0 for each product
        
        Args:
            df (pd.DataFrame): Product data to validate
            
        Returns:
            bool: True if all validation checks pass, False otherwise
        """
        # Check 1: Verify all required columns are present
        # Missing columns would cause downstream calculation failures
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                return False

        # Check 2: Ensure numeric columns contain only numeric data
        # Non-numeric data in these columns would break mathematical operations
        for col in self.NUMERIC_COLUMNS:
            if not pdt.is_numeric_dtype(df[col]):
                return False

        # Check 3: Validate rate columns are numeric and within valid range
        # Rates represent percentages and must be between 0 and 1
        for col in self.RATE_COLUMNS:
            # First check if the column is numeric
            if not pdt.is_numeric_dtype(df[col]):
                return False
            # Then check if all values are within the valid range [0,1]
            if df[col].lt(0).any() or df[col].gt(1).any():
                return False

        # Check 4: Ensure end-of-life rates don't exceed 100% total
        # The sum of recycling, landfill, and incineration rates cannot exceed 1.0
        # Some products might have rates summing to less than 1.0 (indicating other disposal methods)
        if (df[self.RATE_COLUMNS].sum(axis=1) > 1).any():
            return False

        # If all checks pass, the data is valid for LCA analysis
        return True

    def read_impact_factors(self, file_path):
        """
        Load environmental impact factors from a JSON configuration file.
        
        Impact factors define the environmental burden per unit of material for each
        life cycle stage. These factors are essential for converting product quantities
        into environmental impacts. The expected structure is:
        
        {
            "material_name": {
                "life_cycle_stage": {
                    "carbon_impact": factor_value,
                    "energy_impact": factor_value,
                    "water_impact": factor_value
                }
            }
        }
        
        Args:
            file_path (str): Path to JSON file containing impact factors
            
        Returns:
            dict: Nested dictionary structure with impact factors
            
        Raises:
            FileNotFoundError: If impact factors file doesn't exist
            json.JSONDecodeError: If JSON file is malformed
        """
        # Convert to Path object for consistent path handling
        path = Path(file_path)
        
        # Load and parse JSON impact factors
        with open(path, 'r') as f:
            return json.load(f)