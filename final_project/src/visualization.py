# src/visualization.py

"""
LCA Visualization Module

This module provides comprehensive visualization capabilities for Life Cycle Assessment
data analysis. It generates professional-quality charts and plots that help stakeholders
understand environmental impacts across different dimensions such as materials,
lifecycle stages, and product comparisons.

The module uses matplotlib with a non-interactive backend to ensure compatibility
across different environments, including headless servers and testing frameworks.
"""

# Configure matplotlib to use non-interactive backend
# This prevents GUI-related errors in headless environments and during testing
import matplotlib
matplotlib.use('Agg')  # Agg backend works without display/GUI requirements

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LCAVisualizer:
    """
    Comprehensive visualization engine for Life Cycle Assessment data.
    
    This class provides a suite of visualization methods specifically designed
    for environmental impact analysis. Each method creates publication-ready
    charts that highlight different aspects of the LCA results, from detailed
    breakdowns to comparative analyses.
    
    All methods return matplotlib Figure objects, allowing for flexible
    integration into different workflows and output formats.
    """

    def plot_impact_breakdown(self, data: pd.DataFrame, impact_metric, group_by):
        """
        Create a bar chart showing environmental impact breakdown by category.
        
        This visualization helps identify which categories (materials, lifecycle stages,
        etc.) contribute most significantly to a specific environmental impact.
        It's particularly useful for hotspot analysis and identifying areas for
        improvement in product design or process optimization.
        
        Args:
            data (pd.DataFrame): LCA data with calculated impacts
            impact_metric (str): The impact column to analyze (e.g., 'carbon_impact')
            group_by (str): The column to group by (e.g., 'material_type', 'life_cycle_stage')
            
        Returns:
            matplotlib.figure.Figure: Bar chart showing impact breakdown
        """
        # Aggregate impact data by the specified grouping variable
        # Sum all impacts within each group to show total contribution
        df = data.groupby(group_by)[impact_metric].sum().reset_index()
        
        # Create a new figure and axis for the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate bar chart with appropriate styling
        bars = ax.bar(df[group_by], df[impact_metric], 
                     color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Configure axis labels and formatting
        ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(impact_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{impact_metric.replace("_", " ").title()} Breakdown by {group_by.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars for precise reading
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Optimize layout to prevent label cutoff
        plt.tight_layout()
        return fig

    def plot_life_cycle_impacts(self, data: pd.DataFrame, product_id):
        """
        Create a multi-panel chart showing all impact categories across lifecycle stages.
        
        This comprehensive visualization displays how different environmental impacts
        (carbon, energy, water, waste) vary across lifecycle stages for a specific
        product. It uses a 2x2 subplot layout to show all impact types simultaneously,
        enabling holistic analysis of environmental performance.
        
        Args:
            data (pd.DataFrame): LCA data with calculated impacts
            product_id (str): ID of the product to analyze
            
        Returns:
            matplotlib.figure.Figure: Multi-panel chart with lifecycle impacts
        """
        # Filter data to show only the specified product
        df = data[data['product_id'] == product_id]
        
        # Define the impact metrics to display
        metrics = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        stages = df['life_cycle_stage']
        
        # Create a 2x2 subplot layout for comprehensive impact overview
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Convert 2D array to 1D for easier iteration

        # Generate individual bar charts for each impact metric
        for ax, metric in zip(axes, metrics):
            # Create bar chart for this specific impact metric
            bars = ax.bar(stages, df[metric], 
                         color=['lightcoral', 'lightblue', 'lightgreen'][:len(stages)],
                         alpha=0.8, edgecolor='black')
            
            # Format the metric name for display
            metric_display = metric.replace('_', ' ').title()
            ax.set_title(metric_display, fontsize=12, fontweight='bold')
            ax.set_xlabel('Life Cycle Stage', fontsize=10)
            ax.set_ylabel(metric_display, fontsize=10)
            
            # Rotate x-axis labels to prevent overlap
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Add overall title for the entire figure
        fig.suptitle(f'Lifecycle Environmental Impacts - {product_id}', 
                    fontsize=16, fontweight='bold')
        
        # Optimize layout to prevent overlap
        plt.tight_layout()
        return fig

    def plot_product_comparison(self, data: pd.DataFrame, product_ids):
        """
        Create a grouped bar chart comparing environmental impacts between products.
        
        This visualization enables direct comparison of environmental performance
        between alternative products or design options. It displays multiple
        impact categories side-by-side, making it easy to identify which products
        perform better in specific environmental dimensions.
        
        Args:
            data (pd.DataFrame): LCA data with calculated impacts
            product_ids (list): List of product IDs to compare
            
        Returns:
            matplotlib.figure.Figure: Grouped bar chart for product comparison
        """
        # Aggregate total impacts by product across all lifecycle stages
        total = data.groupby('product_id')[['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']].sum()
        
        # Filter to only the products we want to compare
        total = total.loc[product_ids]
        
        # Define impact metrics and set up positioning for grouped bars
        metrics = ['carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']
        x = np.arange(len(metrics))  # X positions for impact categories
        width = 0.35  # Width of individual bars
        
        # Create figure with appropriate size for grouped bars
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define colors for different products
        colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold', 'purple']
        
        # Create grouped bars for each product
        for i, pid in enumerate(product_ids):
            # Offset each product's bars to create grouping effect
            x_offset = x + i * width
            bars = ax.bar(x_offset, total.loc[pid, metrics], width, 
                         label=pid, color=colors[i % len(colors)], 
                         alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Configure x-axis to show impact category names
        ax.set_xticks(x + width * (len(product_ids) - 1) / 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        
        # Add labels and legend
        ax.set_ylabel('Impact Value', fontsize=12)
        ax.set_title('Environmental Impact Comparison Across Products', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Products', loc='upper right')
        
        # Add grid for easier value reading
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

    def plot_end_of_life_breakdown(self, data: pd.DataFrame, product_id):
        """
        Create a pie chart showing end-of-life waste management distribution.
        
        This visualization illustrates how a product's waste is managed at the
        end of its lifecycle, showing the proportions going to recycling,
        landfill, and incineration. This information is crucial for understanding
        the circular economy potential and waste management impacts.
        
        Args:
            data (pd.DataFrame): LCA data with end-of-life information
            product_id (str): ID of the product to analyze
            
        Returns:
            matplotlib.figure.Figure: Pie chart showing waste management breakdown
            
        Raises:
            ValueError: If no end-of-life data exists for the specified product
        """
        # Filter data to end-of-life stage for the specified product
        df = data[
            (data['product_id'] == product_id) &
            (data['life_cycle_stage'].str.lower() == 'end-of-life')
        ]
        
        # Check if end-of-life data exists
        if df.empty:
            raise ValueError(f'No End-of-Life data found for product {product_id}')

        # Extract end-of-life rates from the first (and typically only) matching row
        row = df.iloc[0]
        labels = ['Recycling', 'Landfill', 'Incineration']
        sizes = [row['recycling_rate'], row['landfill_rate'], row['incineration_rate']]
        
        # Define colors for different waste management options
        # Green for recycling (positive), red for landfill (negative), orange for incineration (neutral)
        colors = ['lightgreen', 'lightcoral', 'orange']
        
        # Create pie chart with enhanced visual appeal
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors, explode=(0.05, 0, 0),  # Slightly separate recycling
                                         shadow=True, startangle=90)
        
        # Enhance text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Set title with product information
        ax.set_title(f'End-of-Life Waste Management Breakdown\nProduct: {product_id}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Ensure circular pie chart
        ax.axis('equal')
        
        plt.tight_layout()
        return fig

    def plot_impact_correlation(self, data: pd.DataFrame):
        """
        Create a correlation heatmap showing relationships between impact categories.
        
        This visualization helps identify which environmental impacts tend to
        increase or decrease together. Strong correlations can indicate shared
        underlying causes or trade-offs between different environmental dimensions.
        Understanding these relationships is valuable for strategic decision-making.
        
        Args:
            data (pd.DataFrame): LCA data with calculated impacts
            
        Returns:
            matplotlib.figure.Figure: Correlation heatmap with color-coded matrix
        """
        # Define the impact metrics for correlation analysis
        metrics = ['carbon_impact', 'energy_impact', 'water_impact']
        
        # Calculate correlation matrix between impact categories
        corr = data[metrics].corr()
        
        # Create figure with appropriate size for the correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create color-mapped correlation matrix
        # Use coolwarm colormap: blue for negative correlation, red for positive
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar to show correlation scale
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=12)
        
        # Configure axis ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(metrics)))
        
        # Format metric names for display
        formatted_metrics = [m.replace('_', ' ').title() for m in metrics]
        ax.set_xticklabels(formatted_metrics, rotation=45, ha='right')
        ax.set_yticklabels(formatted_metrics)
        
        # Add correlation values as text annotations
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                # Choose text color based on background intensity
                text_color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha='center', va='center', color=text_color, 
                             fontweight='bold', fontsize=11)
        
        # Set title and formatting
        ax.set_title('Environmental Impact Category Correlations', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig