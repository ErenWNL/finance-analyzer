#!/usr/bin/env python3
"""
This script reverts the seasonal patterns fix by restoring the original implementation
"""

import os

def revert_fix():
    """
    Revert the fix from the AIInsights.jsx file
    """
    file_path = "src/components/AIInsights.jsx"
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
        
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the processSeasonalPatterns function
    start_marker = "// Function to get data from the AIInsights component"
    end_marker = "// Add this near the beginning, after state declarations and before useEffect"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Find the beginning of the line before the function
        line_before = content.rfind("\n", 0, start_idx)
        # Find the end of the line after the function
        line_after = content.find("\n", end_idx) + 1
        
        # Remove the function
        content = content[:line_before] + content[line_after:]
    
    # Restore the original seasonal patterns handling
    current_pattern = "seasonal_patterns: processSeasonalPatterns(),"
    original_pattern = """        seasonal_patterns: (Array.isArray(location.state.data.seasonal_patterns) && location.state.data.seasonal_patterns.length === 0) 
          ? {
              month_spending: {},
              category_seasons: {},
              seasonality_strength: 0,
              year_over_year: { growth: {} }
            } 
          : (location.state.data.seasonal_patterns || {
              month_spending: {},
              category_seasons: {},
              seasonality_strength: 0,
              year_over_year: { growth: {} }
            }),"""
    
    if current_pattern in content:
        content = content.replace(current_pattern, original_pattern)
    
    # Remove the debug log in renderSeasonalTab
    debug_log = "// Debug log\n    console.log('Rendering with seasonal data:', data.seasonal_patterns);"
    if debug_log in content:
        content = content.replace(debug_log, "")
    
    # Write the updated file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully reverted changes in {file_path}")
    print("Please restart your application for the changes to take effect.")
    return True

if __name__ == "__main__":
    revert_fix() 