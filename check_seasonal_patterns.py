import pandas as pd
import numpy as np
from datetime import datetime
import json

# Load the sample transactions data
df = pd.read_csv('public/sample_transactions.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Print basic information about the data
print(f"Loaded {len(df)} transactions")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique months: {df['date'].dt.strftime('%Y-%m').nunique()}")

# Check if we have enough data for seasonal analysis
if len(df) >= 180:
    print("Data meets minimum requirements for seasonal analysis")
else:
    print(f"WARNING: Not enough data for seasonal analysis. Need 180 records, have {len(df)}")

# Monthly totals
monthly_totals = df.groupby(df['date'].dt.month)['amount'].agg(['mean', 'std', 'count'])
print("\nMonthly spending:")
print(monthly_totals)

# Try to calculate seasonality strength
try:
    seasonality_strength = np.std(monthly_totals['mean']) / np.mean(monthly_totals['mean']) * 100
    print(f"\nSeasonality strength: {seasonality_strength:.2f}%")
except Exception as e:
    print(f"Error calculating seasonality strength: {str(e)}")

# Create a simple seasonal patterns object
try:
    # Convert month numbers to names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Enhanced monthly spending data
    month_spending = {}
    for m in monthly_totals.index:
        month_spending[month_names[m-1]] = {
            'mean': float(monthly_totals.loc[m, 'mean']),
            'ci_lower': float(monthly_totals.loc[m, 'mean'] - 1.96 * (monthly_totals.loc[m, 'std'] / np.sqrt(monthly_totals.loc[m, 'count']))),
            'ci_upper': float(monthly_totals.loc[m, 'mean'] + 1.96 * (monthly_totals.loc[m, 'std'] / np.sqrt(monthly_totals.loc[m, 'count']))),
            'confidence': float(monthly_totals.loc[m, 'count'])
        }
    
    # Calculate overall seasonality strength
    seasonality_strength = np.std(list(monthly_totals['mean'])) / np.mean(monthly_totals['mean']) * 100
    
    # Create the results dictionary
    seasonal_patterns = {
        'highest_spending_month': month_names[monthly_totals['mean'].idxmax()-1],
        'lowest_spending_month': month_names[monthly_totals['mean'].idxmin()-1],
        'month_spending': month_spending,
        'quarter_spending': {},  # Simplified for this test
        'category_seasons': {},  # Simplified for this test
        'seasonality_strength': float(seasonality_strength),
        'year_over_year': {
            'growth': {},  # Will be empty since we don't have multi-year data
            'comparison': {}
        }
    }
    
    # Print the result
    print("\nSeasonal patterns object:")
    print(json.dumps(seasonal_patterns, indent=2))
except Exception as e:
    print(f"Error creating seasonal patterns: {str(e)}")
    import traceback
    traceback.print_exc() 