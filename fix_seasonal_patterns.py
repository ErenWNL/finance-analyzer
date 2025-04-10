#!/usr/bin/env python3
"""
This script directly applies a fix to the seasonal patterns issue by
modifying the AIInsights.jsx component to handle an empty array properly.
"""

import json
import os

# Define the fix for AIInsights.jsx
FIX_CODE = '''
  // Function to get data from the AIInsights component
  const processSeasonalPatterns = () => {
    // Get the seasonal patterns from the state
    const patterns = location.state?.data?.seasonal_patterns;
    console.log("Processing seasonal patterns:", patterns);

    // If it's an empty array, create our own data structure
    if (Array.isArray(patterns) && patterns.length === 0) {
      console.log("Converting empty array to proper seasonal patterns object");
      
      // Create month data
      const monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
      ];
      const monthSpending = {};
      monthNames.forEach(month => {
        monthSpending[month] = {
          mean: 0.0,
          ci_lower: 0.0,
          ci_upper: 0.0,
          confidence: 0.0
        };
      });
      
      // Create properly structured object
      const seasonalObj = {
        highest_spending_month: monthNames[0],
        lowest_spending_month: monthNames[0],
        month_spending: monthSpending,
        quarter_spending: {
          Q1: { mean: 0.0, trend: 0.0 },
          Q2: { mean: 0.0, trend: 0.0 },
          Q3: { mean: 0.0, trend: 0.0 },
          Q4: { mean: 0.0, trend: 0.0 }
        },
        category_seasons: {},
        seasonality_strength: 0.0,
        year_over_year: {
          growth: {},
          comparison: {}
        }
      };
      
      // Return the fixed object
      return seasonalObj;
    }
    
    // If it's already a valid object (not an array), just return it
    return patterns || {};
  };

  // Add this near the beginning, after state declarations and before useEffect
'''

def apply_fix():
    """
    Apply the fix to the AIInsights.jsx file
    """
    file_path = "src/components/AIInsights.jsx"
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
        
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find a good spot to insert our function
    target_line = "const formatCurrency = (amount) => {"
    
    if target_line not in content:
        print(f"Couldn't find insertion point: {target_line}")
        return False
    
    # Insert the function right before the formatCurrency function
    new_content = content.replace(target_line, FIX_CODE + "\n  " + target_line)
    
    # Also update the seasonal patterns in useEffect
    pattern_line = "seasonal_patterns: (Array.isArray(location.state.data.seasonal_patterns) && location.state.data.seasonal_patterns.length === 0) "
    replacement = "seasonal_patterns: processSeasonalPatterns(),"
    
    if pattern_line in new_content:
        # Replace the entire section with our simpler call
        start_idx = new_content.find(pattern_line)
        end_idx = new_content.find("models_trained", start_idx)
        
        if start_idx != -1 and end_idx != -1:
            line_before = new_content[:start_idx].rstrip()
            line_after = new_content[end_idx:]
            new_content = line_before + "\n        " + replacement + "\n        " + line_after
    
    # And another fix in renderSeasonalTab function
    render_line = "const renderSeasonalTab = () => {"
    if render_line in new_content:
        # Find the start of the function
        start_idx = new_content.find(render_line)
        # Find the beginning of the return statement
        return_idx = new_content.find("return", start_idx)
        
        if start_idx != -1 and return_idx != -1:
            # Insert logging
            debug_log = "\n    // Debug log\n    console.log('Rendering with seasonal data:', data.seasonal_patterns);\n"
            
            # Insert right before the return
            new_content = new_content[:return_idx] + debug_log + new_content[return_idx:]
    
    # Write the updated file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully applied fix to {file_path}")
    print("Please restart your application for the changes to take effect.")
    return True

if __name__ == "__main__":
    apply_fix() 