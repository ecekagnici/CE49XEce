# Life Cycle Assessment (LCA) Analysis Tool

## 🌍 Project Overview

This comprehensive Life Cycle Assessment (LCA) tool is designed to analyze and visualize the environmental impacts of products throughout their entire lifecycle. The tool integrates advanced data science methodologies with environmental science principles to provide detailed insights into carbon footprint, energy consumption, water usage, and waste generation across different lifecycle stages.

### Key Features

- **Multi-format Data Support**: Seamlessly handles CSV, Excel, and JSON data formats
- **Comprehensive Impact Analysis**: Calculates carbon, energy, and water impacts with detailed lifecycle tracking
- **Advanced Visualizations**: Generates professional charts for impact breakdown, product comparison, and correlation analysis
- **Robust Data Validation**: Ensures data quality and completeness before analysis
- **Command-line Interface**: Professional CLI for automated analysis pipelines
- **Comparative Analysis**: Side-by-side product comparison with relative impact metrics
- **End-of-life Analysis**: Detailed waste management and disposal impact assessment

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd lca-analysis-tool
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation by running tests:**
```bash
python -m pytest
```

### Basic Usage

Run a complete LCA analysis with a single command:

```bash
python src/main.py --data data/raw/sample_data.csv --factors data/raw/impact_factors.json --output my_analysis
```

## 📁 Project Structure

```
lca-analysis-tool/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # Command-line interface
│   ├── data_input.py                 # Data loading and validation
│   ├── calculations.py               # LCA impact calculations
│   ├── visualization.py              # Chart generation
│   └── utils.py                      # Utility functions
├── tests/                            # Test suite
│   ├── conftest.py                   # Test configuration
│   ├── test_data_input.py           # Data handling tests
│   ├── test_calculations.py         # Calculation tests
│   └── test_visualization.py        # Visualization tests
└── data/                            # Data directory
    ├── raw/                         # Input data files
    │   ├── sample_data.csv          # Example product data
    │   └── impact_factors.json      # Environmental impact factors
    └── results/                     # Output directory (created automatically)
```

## 💻 Command Line Usage

### Main Analysis Pipeline

The primary interface is through the `main.py` script, which orchestrates the complete LCA workflow:

```bash
python src/main.py [OPTIONS]
```

#### Required Arguments:
- `--data, -d`: Path to product data file (CSV/Excel/JSON)
- `--factors, -f`: Path to impact factors JSON file

#### Optional Arguments:
- `--output, -o`: Output directory name (default: "results")

#### Example Commands:

**Basic analysis:**
```bash
python src/main.py -d data/raw/sample_data.csv -f data/raw/impact_factors.json
```

**Custom output directory:**
```bash
python src/main.py --data data/raw/products.xlsx --factors data/raw/impacts.json --output quarterly_analysis
```

**Full command with all options:**
```bash
python src/main.py \
  --data data/raw/manufacturing_data.csv \
  --factors data/raw/updated_impact_factors.json \
  --output detailed_analysis_2024
```

### What Happens When You Run the Analysis:

1. **Data Loading**: Reads and validates your product data
2. **Impact Calculation**: Computes environmental impacts using impact factors
3. **Report Generation**: Creates CSV files with detailed results
4. **Visualization**: Generates comprehensive charts and graphs
5. **Output Organization**: Saves everything to the specified output directory

## 📊 Output Files and Results

### Output Directory Structure

When you run the analysis, results are automatically organized in the `data/` directory:

```
data/
├── raw/                              # Your input files
└── [results]/                    # Results directory (e.g., "results" or "my_analysis")
    ├── detailed_impacts.csv          # Impact calculations per product/stage
    ├── total_impacts.csv             # Aggregated impacts by product
    ├── normalized_impacts.csv        # Normalized values for comparison
    ├── comparison.csv                # Product comparison (if multiple products)
    ├── breakdown_by_material_type.png # Material impact breakdown chart
    ├── lifecycle_[ProductID].png     # Lifecycle analysis for first product
    ├── product_comparison.png        # Product comparison chart
    ├── end_of_life_[ProductID].png   # End-of-life breakdown chart
    └── impact_correlation.png        # Impact correlation heatmap
```

### Generated Files Explained:

#### CSV Reports:
- **`detailed_impacts.csv`**: Complete breakdown showing calculated impacts for each product at every lifecycle stage
- **`total_impacts.csv`**: Summary showing total environmental impacts aggregated by product
- **`normalized_impacts.csv`**: Scaled values (0-1 range) for easy comparison between different impact types
- **`comparison.csv`**: Relative comparison between products (if multiple products exist)

#### Visualization Charts:
- **Material Breakdown**: Bar chart showing which materials contribute most to environmental impacts
- **Lifecycle Analysis**: Multi-panel view showing how impacts vary across lifecycle stages
- **Product Comparison**: Side-by-side comparison of environmental performance
- **End-of-life Analysis**: Pie chart showing waste management distribution (recycling, landfill, incineration)
- **Impact Correlation**: Heatmap revealing relationships between different environmental impacts

### Example Output Locations:

If you run:
```bash
python src/main.py -d data/raw/sample_data.csv -f data/raw/impact_factors.json -o my_results
```

Your results will be saved in: `data/my_results/`

## 🧪 Testing

### Running the Test Suite

**Run all tests:**
```bash
python -m pytest
```

**Run tests with verbose output:**
```bash
python -m pytest -v
```

**Run specific test files:**
```bash
# Test data input functionality
python -m pytest tests/test_data_input.py -v

# Test calculation functions
python -m pytest tests/test_calculations.py -v

# Test visualization generation
python -m pytest tests/test_visualization.py -v
```

**Run tests with coverage report:**
```bash
python -m pytest --cov=src
```

### Test Categories:

- **Data Input Tests**: Verify file reading, validation, and error handling
- **Calculation Tests**: Ensure accurate impact calculations and aggregations
- **Visualization Tests**: Confirm chart generation and proper formatting

## 📋 Data Format Requirements

### Product Data File Structure

Your product data file must include these columns:

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `product_id` | String | Unique product identifier | "P001" |
| `product_name` | String | Product name | "Steel Frame" |
| `life_cycle_stage` | String | Lifecycle stage | "Manufacturing" |
| `material_type` | String | Material type | "steel" |
| `quantity_kg` | Numeric | Quantity in kilograms | 100.5 |
| `energy_consumption_kwh` | Numeric | Energy consumption | 120.0 |
| `transport_distance_km` | Numeric | Transportation distance | 50.0 |
| `transport_mode` | String | Transport method | "Truck" |
| `waste_generated_kg` | Numeric | Waste generated | 5.2 |
| `recycling_rate` | Numeric (0-1) | Recycling percentage | 0.85 |
| `landfill_rate` | Numeric (0-1) | Landfill percentage | 0.10 |
| `incineration_rate` | Numeric (0-1) | Incineration percentage | 0.05 |
| `carbon_footprint_kg_co2e` | Numeric | Carbon footprint | 180.5 |
| `water_usage_liters` | Numeric | Water usage | 150.0 |

### Impact Factors File Structure

The impact factors JSON file should follow this structure:

```json
{
  "material_name": {
    "lifecycle_stage": {
      "carbon_impact": numeric_value,
      "energy_impact": numeric_value,
      "water_impact": numeric_value
    }
  }
}
```

**Example:**
```json
{
  "steel": {
    "manufacturing": {
      "carbon_impact": 1.8,
      "energy_impact": 20,
      "water_impact": 150
    },
    "transportation": {
      "carbon_impact": 0.5,
      "energy_impact": 5,
      "water_impact": 30
    },
    "end-of-life": {
      "carbon_impact": 0.1,
      "energy_impact": 1,
      "water_impact": 10
    }
  }
}
```

## 🔧 Advanced Usage

### Programmatic Usage

You can also use the LCA tool components directly in your Python code:

```python
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer

# Load data
data_input = DataInput()
product_data = data_input.read_data('data/raw/sample_data.csv')

# Validate data
if data_input.validate_data(product_data):
    # Calculate impacts
    calculator = LCACalculator('data/raw/impact_factors.json')
    impacts = calculator.calculate_impacts(product_data)
    
    # Generate visualizations
    visualizer = LCAVisualizer()
    fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', 'material_type')
    fig.savefig('my_custom_chart.png')
```

## 📈 Interpreting Results

### Key Metrics:
- **Carbon Impact**: CO2 equivalent emissions per unit
- **Energy Impact**: Energy consumption in standardized units
- **Water Impact**: Water usage and pollution potential
- **Waste Generation**: Physical waste production

### Comparison Guidelines:
- Lower values generally indicate better environmental performance
- Use normalized values for comparing different impact types
- Consider relative values when comparing products
- Pay attention to lifecycle stage contributions for optimization opportunities

## 🐛 Troubleshooting

### Common Issues:

**Import Errors:**
```bash
# Use python -m pytest instead of just pytest
python -m pytest

# For main script, ensure you're in the project root
python src/main.py -d data/raw/sample_data.csv -f data/raw/impact_factors.json
```

**File Not Found:**
```bash
# Check file paths are correct relative to project root
ls data/raw/  # Should show your data files
```

**Data Validation Errors:**
- Ensure all required columns are present
- Check that numeric columns contain only numbers
- Verify end-of-life rates sum to ≤ 1.0

## 📚 Dependencies

The tool requires these Python packages (see `requirements.txt`):

- `pandas>=1.5.0` - Data manipulation and analysis
- `matplotlib>=3.5.0` - Visualization and plotting
- `numpy>=1.21.0` - Numerical computations
- `pytest>=7.0.0` - Testing framework
- `openpyxl>=3.0.0` - Excel file support