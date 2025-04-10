#!/usr/bin/env python3
"""
This script implements real seasonal patterns calculation in the AIInsights component
"""

import os

# Define the implementation code
IMPLEMENTATION_CODE = '''
  // Function to calculate real seasonal patterns using transaction data
  const calculateSeasonalPatterns = (transactions) => {
    console.log("Calculating real seasonal patterns from", transactions.length, "transactions");
    
    // Return default if no transactions
    if (!transactions || transactions.length < 180) {
      console.log("Not enough data for seasonal patterns (need 180+ transactions)");
      return {
        highest_spending_month: "January",
        lowest_spending_month: "January",
        month_spending: {},
        quarter_spending: {},
        category_seasons: {},
        seasonality_strength: 0,
        year_over_year: { growth: {}, comparison: {} }
      };
    }
    
    try {
      // Group transactions by month
      const monthlyData = {};
      const monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
      ];
      
      // Initialize monthly data
      monthNames.forEach((month, index) => {
        monthlyData[month] = {
          transactions: [],
          amount: 0,
          count: 0,
          mean: 0,
          values: []
        };
      });
      
      // Group transactions by month
      transactions.forEach(tx => {
        if (!tx.date || !tx.amount) return;
        
        const date = new Date(tx.date);
        const monthIndex = date.getMonth();
        const monthName = monthNames[monthIndex];
        const amount = parseFloat(tx.amount);
        
        if (isNaN(amount)) return;
        
        monthlyData[monthName].transactions.push(tx);
        monthlyData[monthName].amount += amount;
        monthlyData[monthName].count++;
        monthlyData[monthName].values.push(amount);
      });
      
      // Calculate monthly statistics
      let highestMonth = monthNames[0];
      let lowestMonth = monthNames[0];
      let highestAmount = 0;
      let lowestAmount = Number.MAX_VALUE;
      
      const means = [];
      
      monthNames.forEach(month => {
        const data = monthlyData[month];
        if (data.count > 0) {
          data.mean = data.amount / data.count;
          
          // Calculate standard deviation
          const sumSquares = data.values.reduce((sum, val) => sum + Math.pow(val - data.mean, 2), 0);
          data.std = Math.sqrt(sumSquares / data.count);
          
          // Keep track of highest/lowest months
          if (data.mean > highestAmount) {
            highestAmount = data.mean;
            highestMonth = month;
          }
          
          if (data.mean < lowestAmount && data.count > 0) {
            lowestAmount = data.mean;
            lowestMonth = month;
          }
          
          means.push(data.mean);
        }
      });
      
      // Format the month spending data for the component
      const monthSpending = {};
      monthNames.forEach(month => {
        const data = monthlyData[month];
        if (data.count > 0) {
          // Calculate confidence interval
          const ciMargin = 1.96 * (data.std / Math.sqrt(data.count));
          
          monthSpending[month] = {
            mean: data.mean,
            ci_lower: data.mean - ciMargin,
            ci_upper: data.mean + ciMargin,
            confidence: data.count
          };
        } else {
          monthSpending[month] = {
            mean: 0,
            ci_lower: 0,
            ci_upper: 0,
            confidence: 0
          };
        }
      });
      
      // Calculate seasonality strength
      if (means.length > 0) {
        const meanOfMeans = means.reduce((sum, val) => sum + val, 0) / means.length;
        const sumSquaredDiffs = means.reduce((sum, val) => sum + Math.pow(val - meanOfMeans, 2), 0);
        const stdOfMeans = Math.sqrt(sumSquaredDiffs / means.length);
        const seasonalityStrength = (stdOfMeans / meanOfMeans) * 100;
        
        // Create quarter data
        const quarterSpending = {
          Q1: { mean: 0, trend: 0 },
          Q2: { mean: 0, trend: 0 },
          Q3: { mean: 0, trend: 0 },
          Q4: { mean: 0, trend: 0 }
        };
        
        // Calculate quarterly averages
        const q1Months = ['January', 'February', 'March'];
        const q2Months = ['April', 'May', 'June'];
        const q3Months = ['July', 'August', 'September'];
        const q4Months = ['October', 'November', 'December'];
        
        [
          { quarter: 'Q1', months: q1Months },
          { quarter: 'Q2', months: q2Months },
          { quarter: 'Q3', months: q3Months },
          { quarter: 'Q4', months: q4Months }
        ].forEach(q => {
          let sum = 0;
          let count = 0;
          
          q.months.forEach(month => {
            if (monthlyData[month].count > 0) {
              sum += monthlyData[month].mean;
              count++;
            }
          });
          
          quarterSpending[q.quarter].mean = count > 0 ? sum / count : 0;
        });
        
        // Create category analysis
        const categorySeasons = {};
        const categories = {};
        
        // Group by category
        transactions.forEach(tx => {
          if (!tx.category) return;
          
          if (!categories[tx.category]) {
            categories[tx.category] = {
              transactions: [],
              monthlyData: {}
            };
            
            // Initialize monthly data for category
            monthNames.forEach(month => {
              categories[tx.category].monthlyData[month] = {
                amount: 0,
                count: 0,
                mean: 0
              };
            });
          }
          
          categories[tx.category].transactions.push(tx);
          
          // Group by month
          if (tx.date) {
            const date = new Date(tx.date);
            const monthName = monthNames[date.getMonth()];
            
            if (parseFloat(tx.amount)) {
              categories[tx.category].monthlyData[monthName].amount += parseFloat(tx.amount);
              categories[tx.category].monthlyData[monthName].count++;
            }
          }
        });
        
        // Calculate monthly means for each category
        Object.keys(categories).forEach(category => {
          if (categories[category].transactions.length < 20) return;
          
          const catData = categories[category];
          
          // Calculate monthly means
          monthNames.forEach(month => {
            const monthData = catData.monthlyData[month];
            monthData.mean = monthData.count > 0 ? monthData.amount / monthData.count : 0;
          });
          
          // Find peak and low months
          let peakMonth = monthNames[0];
          let lowMonth = monthNames[0];
          let peakMean = 0;
          let lowMean = Number.MAX_VALUE;
          
          monthNames.forEach(month => {
            const mean = catData.monthlyData[month].mean;
            
            if (mean > peakMean && catData.monthlyData[month].count > 0) {
              peakMean = mean;
              peakMonth = month;
            }
            
            if (mean < lowMean && catData.monthlyData[month].count > 0) {
              lowMean = mean;
              lowMonth = month;
            }
          });
          
          categorySeasons[category] = {
            peak_month: peakMonth,
            low_month: lowMonth,
            peak_spending: peakMean,
            low_spending: lowMean,
            seasonality_index: {}
          };
        });
        
        // Return the full patterns object
        return {
          highest_spending_month: highestMonth,
          lowest_spending_month: lowestMonth,
          month_spending: monthSpending,
          quarter_spending: quarterSpending,
          category_seasons: categorySeasons,
          seasonality_strength: seasonalityStrength,
          year_over_year: { growth: {}, comparison: {} } // Simplified
        };
      }
    } catch (error) {
      console.error("Error calculating seasonal patterns:", error);
    }
    
    // Return default on error
    return {
      highest_spending_month: "January",
      lowest_spending_month: "January",
      month_spending: {},
      quarter_spending: {},
      category_seasons: {},
      seasonality_strength: 0,
      year_over_year: { growth: {}, comparison: {} }
    };
  };
'''

def implement_seasonal_patterns():
    """
    Implement the real seasonal patterns calculation in AIInsights.jsx
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
    new_content = content.replace(target_line, IMPLEMENTATION_CODE + "\n  " + target_line)
    
    # Update the seasonal patterns processing in useEffect
    original_pattern = """seasonal_patterns: (Array.isArray(location.state.data.seasonal_patterns) && location.state.data.seasonal_patterns.length === 0) 
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
    
    replacement = "seasonal_patterns: calculateSeasonalPatterns(location.state.data.expenses || []),"
    
    if original_pattern in new_content:
        new_content = new_content.replace(original_pattern, replacement)
    
    # Add debug log to the renderSeasonalTab function
    render_line = "const renderSeasonalTab = () => {"
    debug_log = "\n    // Debug log\n    console.log('Rendering with calculated seasonal data:', data.seasonal_patterns);\n"
    
    if render_line in new_content:
        # Find the start of the function
        start_idx = new_content.find(render_line)
        # Find the beginning of the return statement
        return_idx = new_content.find("return", start_idx)
        
        if start_idx != -1 and return_idx != -1 and "Rendering with calculated seasonal data" not in new_content:
            # Insert right before the return
            new_content = new_content[:return_idx] + debug_log + new_content[return_idx:]
    
    # Write the updated file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully implemented real seasonal patterns calculation in {file_path}")
    print("Please restart your application for the changes to take effect.")
    return True

if __name__ == "__main__":
    implement_seasonal_patterns() 