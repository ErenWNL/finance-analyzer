import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Button,
  CircularProgress,
  Box,
  Paper,
  Divider,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  ArrowLeft,
  BrainCircuit,
  Calendar,
  Activity,
  LineChart,
  DollarSign
} from 'lucide-react';
import PredictionsTab from './PredictionsTab';
import BudgetRecommendations from './BudgetRecommendations';
import AnomalyDetection from './AnomalyDetection';
import TransactionCategorizer from './TransactionCategorizer';

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
    
    // Log monthly totals
    console.log("Monthly transaction counts:", Object.entries(monthlyData).map(([month, data]) => ({
      month,
      count: data.count,
      total: data.amount
    })));
    
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
    
    // Log monthly means
    console.log("Monthly means:", Object.entries(monthlyData).map(([month, data]) => ({
      month,
      mean: data.mean,
      std: data.std
    })));
    
    // Format the month spending data
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
    const meanOfMeans = means.reduce((sum, val) => sum + val, 0) / means.length;
    const sumSquaredDiffs = means.reduce((sum, val) => sum + Math.pow(val - meanOfMeans, 2), 0);
    const stdOfMeans = Math.sqrt(sumSquaredDiffs / means.length);
    const seasonalityStrength = (stdOfMeans / meanOfMeans) * 100;
    
    console.log("Seasonality calculation:", {
      meanOfMeans,
      stdOfMeans,
      seasonalityStrength
    });
    
    // Create quarter data
    const quarterSpending = {
      Q1: { mean: 0, trend: 0 },
      Q2: { mean: 0, trend: 0 },
      Q3: { mean: 0, trend: 0 },
      Q4: { mean: 0, trend: 0 }
    };
    
    // Calculate quarterly averages
    [
      { quarter: 'Q1', months: ['January', 'February', 'March'] },
      { quarter: 'Q2', months: ['April', 'May', 'June'] },
      { quarter: 'Q3', months: ['July', 'August', 'September'] },
      { quarter: 'Q4', months: ['October', 'November', 'December'] }
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
    
    console.log("Quarterly spending:", quarterSpending);
    
    // Return the full patterns object
    return {
      highest_spending_month: highestMonth,
      lowest_spending_month: lowestMonth,
      month_spending: monthSpending,
      quarter_spending: quarterSpending,
      category_seasons: {}, // Simplified for now
      seasonality_strength: seasonalityStrength,
      year_over_year: { growth: {}, comparison: {} } // Simplified
    };
  } catch (error) {
    console.error("Error calculating seasonal patterns:", error);
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
};

const AIInsights = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  
  // currency formatting
  
  
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(amount);
  };

  useEffect(() => {
    // Check if data exists in location state
    if (location.state?.data) {
      console.log('Received AI insights data:', location.state.data);
      
      // Check if expenses were passed correctly
      if (location.state.data.expenses) {
        console.log(`Received ${location.state.data.expenses.length} expenses`);
        
        // Check for uncategorized transactions
        const uncategorized = location.state.data.expenses.filter(
          tx => !tx.category || tx.category === '' || 
                tx.category === 'Uncategorized' || 
                tx.category === 'Other'
        );
        console.log(`Found ${uncategorized.length} uncategorized expenses`, uncategorized);
      } else {
        console.warn('No expenses array in data');
      }
      
      // Debug seasonal patterns
      console.log('Seasonal patterns received:', location.state.data.seasonal_patterns);
      console.log('Seasonal patterns type:', typeof location.state.data.seasonal_patterns);
      console.log('Is Array?', Array.isArray(location.state.data.seasonal_patterns));
      console.log('Seasonal patterns keys:', Object.keys(location.state.data.seasonal_patterns || {}));
      
      // Create default seasonal data structure
      const defaultSeasonalData = {
        highest_spending_month: "January",
        lowest_spending_month: "January",
        month_spending: {},
        quarter_spending: {},
        category_seasons: {},
        seasonality_strength: 0,
        year_over_year: { growth: {}, comparison: {} }
      };
      
      // Helper function to check if seasonal patterns is valid
      const isValidSeasonalPatterns = (data) => {
        const isValid = data && 
               typeof data === 'object' && 
               !Array.isArray(data) && 
               Object.keys(data).length > 0;
        console.log('Is seasonal patterns valid?', isValid);
        return isValid;
      };
      
      // Convert array to object if needed and ensure all required fields are present
      let processedSeasonalPatterns = defaultSeasonalData;
      
      if (location.state.data.seasonal_patterns) {
        if (Array.isArray(location.state.data.seasonal_patterns)) {
          console.log('Converting array to seasonal patterns object');
          // If it's an array, calculate seasonal patterns from expenses data
          processedSeasonalPatterns = calculateSeasonalPatterns(location.state.data.expenses || []);
        } else if (isValidSeasonalPatterns(location.state.data.seasonal_patterns)) {
          console.log('Merging with default seasonal data');
          processedSeasonalPatterns = {
            ...defaultSeasonalData,
            ...location.state.data.seasonal_patterns
          };
        }
      }
      
      console.log('Final processed seasonal patterns:', processedSeasonalPatterns);
      
      // Ensure all required fields are present
      const processedData = {
        ai_insights: location.state.data.ai_insights || {},
        transaction_count: location.state.data.transaction_count || 0,
        spending_insights: location.state.data.spending_insights || [],
        spending_velocity: location.state.data.spending_velocity || {},
        future_predictions: location.state.data.future_predictions || [],
        anomalous_transactions: location.state.data.anomalous_transactions || [],
        seasonal_patterns: processedSeasonalPatterns,
        models_trained: location.state.data.models_trained || false,
        expenses: location.state.data.expenses || [] // Store expenses in data
      };
      
      // Debug the processed seasonal patterns
      console.log('Processed seasonal patterns:', processedData.seasonal_patterns);
      
      setData(processedData);
      setLoading(false);
    } else {
      console.error('No insights data available in location state');
      setError("No insights data available. Please analyze your expenses first.");
      setLoading(false);
    }
  }, [location.state]);

  const handleBack = () => {
    navigate('/dashboard');
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Add this handler for when transactions are categorized
  const handleCategorizedTransactions = (updatedTransactions) => {
    if (!updatedTransactions || updatedTransactions.length === 0) {
      console.error('No transactions received from categorizer');
      return;
    }
    
    console.log(`Received ${updatedTransactions.length} transactions from categorizer`);
    
    // Calculate how many were actually categorized (for user feedback)
    const categorizedCount = updatedTransactions.filter(tx => 
      tx.category && tx.category !== 'Uncategorized' && tx.category !== 'Other'
    ).length;
    
    console.log(`${categorizedCount} transactions have categories other than Uncategorized/Other`);
    
    // Navigate back to dashboard with the updated transactions
    navigate('/dashboard', { 
      state: { 
        categorizedTransactions: updatedTransactions,
        message: `Successfully categorized ${categorizedCount} transactions`
      }
    });
  };
  

  // Show loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 p-8 flex justify-center items-center">
        <Card className="max-w-md w-full">
          <CardContent className="flex flex-col items-center p-8">
            <CircularProgress className="mb-4" />
            <Typography variant="h6">Loading AI insights...</Typography>
          </CardContent>
        </Card>
      </div>
    );
  }

  // If data is not available or there's an error
  if (error || !data) {
    return (
      <div className="min-h-screen bg-gray-100 p-8">
        <Card className="max-w-md mx-auto">
          <CardContent>
            <div className="flex flex-col items-center gap-4">
              <AlertCircle className="text-red-500 w-12 h-12" />
              <Typography variant="h6" color="error">
                {error || "No insights available"}
              </Typography>
              <Typography variant="body2" className="text-center mb-4">
                {error === "No insights data available. Please analyze your expenses first." 
                  ? "Please go back to the dashboard and analyze your expenses to view AI insights."
                  : "We couldn't find any AI insights for your expenses. Please make sure you have enough transaction data for analysis."}
              </Typography>
              <Button
                variant="contained"
                startIcon={<ArrowLeft />}
                onClick={handleBack}
              >
                Return to Dashboard
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Extract insights from data
  const insights = data.ai_insights || {};
  const transactionCount = data.transaction_count || 0;

  // Check if prediction models are trained
  const modelsAvailable = insights.models_trained || 
                       (insights.future_predictions && insights.future_predictions.length > 0);

  // Components for each tab
  const renderInsightsTab = () => (
    <div className="space-y-6">
      {/* Spending Insights Section */}
      {insights.spending_insights?.length > 0 && (
        <Card className="border-l-4 border-green-500">
          <CardContent className="p-6">
            <Typography variant="h6" className="text-green-700 mb-3 flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Key Spending Insights
            </Typography>
            <List>
              {insights.spending_insights.map((insight, index) => (
                <ListItem key={index} className="py-2">
                  <ListItemIcon>
                    {insight.type === 'trend' ? (
                      insight.message.includes('increasing') ? (
                        <TrendingUp className="text-red-500" />
                      ) : (
                        <TrendingDown className="text-green-500" />
                      )
                    ) : insight.type === 'prediction' ? (
                      <LineChart className="text-blue-500" />
                    ) : insight.type === 'monthly_change' ? (
                      insight.message.includes('increased') ? (
                        <TrendingUp className="text-red-500" />
                      ) : (
                        <TrendingDown className="text-green-500" />
                      )
                    ) : (
                      <AlertCircle className="text-blue-500" />
                    )}
                  </ListItemIcon>
                  <ListItemText 
                    primary={insight.message}
                    className={
                      insight.type === 'trend' && insight.message.includes('increasing')
                        ? 'text-red-600'
                        : insight.type === 'trend' ? 'text-green-600' 
                        : insight.type === 'monthly_change' && insight.message.includes('increased')
                        ? 'text-red-600'
                        : insight.type === 'monthly_change' ? 'text-green-600'
                        : 'text-blue-600'
                    }
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Spending Velocity Section */}
      {insights.spending_velocity !== undefined && Math.abs(insights.spending_velocity) > 3 && (
        <Card className="border-l-4 border-blue-500">
          <CardContent className="p-6">
            <Typography variant="h6" className="text-blue-700 mb-3 flex items-center">
              {insights.spending_velocity > 0 ? (
                <TrendingUp className="w-5 h-5 mr-2 text-red-500" />
              ) : (
                <TrendingDown className="w-5 h-5 mr-2 text-green-500" />
              )}
              Spending Velocity
            </Typography>
            <Typography variant="body1" className="mb-3">
              Your spending is {insights.spending_velocity > 0 ? 'accelerating' : 'decelerating'} at a rate of{' '}
              <span className={insights.spending_velocity > 0 ? 'text-red-600 font-bold' : 'text-green-600 font-bold'}>
                {Math.abs(insights.spending_velocity).toFixed(1)}%
              </span>{' '}
              per month.
            </Typography>
            
            <Box className="p-4 bg-gray-50 rounded-lg">
              <Typography variant="subtitle2" className="mb-2">
                What does this mean?
              </Typography>
              <Typography variant="body2" className="text-gray-700">
                Spending velocity measures how quickly your expenses are changing from month to month. 
                {insights.spending_velocity > 5 
                  ? ' Your rapid increase in spending may impact your financial health if sustained.' 
                  : insights.spending_velocity < -5
                  ? ' Your spending reduction shows good expense management.' 
                  : ' Your spending rate is changing at a moderate pace.'}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Next Month Prediction Section */}
      {(insights.future_predictions?.length > 0 || insights.next_month_prediction) && (
        <Card className="border-l-4 border-purple-500">
          <CardContent className="p-6">
            <Typography variant="h6" className="text-purple-700 mb-3 flex items-center">
              <LineChart className="w-5 h-5 mr-2" />
              Next Month's Prediction
            </Typography>
            
            <div className="flex flex-col md:flex-row md:justify-between">
              {insights.future_predictions?.length > 0 ? (
                <div className="md:w-1/2 space-y-3">
                  <Typography variant="body1">
                    Based on our ML models, your predicted spending for next month is:
                  </Typography>
                  <Box className="flex items-center">
                    <Chip
                      label={formatCurrency(insights.future_predictions[0].total_predicted)}
                      color="primary"
                      className="text-lg px-2 py-4"
                    />
                    
                    {/* Show change from current month if available */}
                    {data.monthly_totals && Object.keys(data.monthly_totals).length > 0 && (
                      (() => {
                        const months = Object.keys(data.monthly_totals).sort();
                        const currentMonth = months[months.length - 1];
                        const currentSpending = data.monthly_totals[currentMonth];
                        const predictedSpending = insights.future_predictions[0].total_predicted;
                        const percentChange = ((predictedSpending - currentSpending) / currentSpending) * 100;
                        
                        if (!isNaN(percentChange)) {
                          return (
                            <Chip
                              icon={percentChange > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                              label={`${percentChange > 0 ? '+' : ''}${percentChange.toFixed(1)}%`}
                              color={percentChange > 5 ? "error" : percentChange < -5 ? "success" : "default"}
                              variant="outlined"
                              size="small"
                              className="ml-2"
                            />
                          );
                        }
                        return null;
                      })()
                    )}
                  </Box>
                </div>
              ) : (
                <div className="md:w-1/2">
                  <Typography variant="body1" className="mb-2">
                    Estimated spending for next month:
                  </Typography>
                  <Chip
                    label={formatCurrency(insights.next_month_prediction || 0)}
                    color="primary"
                    className="text-lg"
                  />
                </div>
              )}
              
              {/* Category Breakdown */}
              {insights.future_predictions?.length > 0 && insights.future_predictions[0].category_predictions && (
                <div className="md:w-1/2 mt-4 md:mt-0">
                  <Typography variant="subtitle2" className="mb-2">
                    Top Category Predictions:
                  </Typography>
                  <List dense>
                    {Object.entries(insights.future_predictions[0].category_predictions)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([category, amount], index) => (
                        <ListItem key={index} disableGutters className="py-1">
                          <ListItemText 
                            primary={`${category}: ${formatCurrency(amount)}`}
                          />
                        </ListItem>
                      ))
                    }
                  </List>
                </div>
              )}
            </div>
            
            {modelsAvailable && (
              <Box className="mt-4 text-center">
                <Button
                  variant="outlined"
                  color="primary"
                  size="small"
                  onClick={() => setActiveTab(1)}
                >
                  View Detailed Predictions
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* ML Model Status */}
      <Card>
        <CardContent className="p-6">
          <div className="flex justify-between items-center mb-4">
            <Typography variant="h6" className="flex items-center">
              <BrainCircuit className="w-5 h-5 mr-2" />
              AI Model Status
            </Typography>
            <Chip
              label={transactionCount > 50 ? 'High Confidence' : transactionCount > 20 ? 'Medium Confidence' : 'Low Confidence'}
              color={transactionCount > 50 ? "success" : transactionCount > 20 ? "primary" : "warning"}
              variant="outlined"
              size="small"
            />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Typography variant="body2">Transaction Count:</Typography>
              <Typography variant="body1" className="font-medium">{transactionCount}</Typography>
            </div>
            <div className="flex justify-between items-center">
              <Typography variant="body2">Prediction Models:</Typography>
              <Chip 
                label={modelsAvailable ? "Trained" : "Not Available"} 
                color={modelsAvailable ? "success" : "default"}
                size="small"
              />
            </div>
            <div className="flex justify-between items-center">
              <Typography variant="body2">Anomaly Detection:</Typography>
              <Chip 
                label="Active" 
                color="success"
                size="small"
              />
            </div>
          </div>
          
          <Divider className="my-4" />
          
          <Typography variant="body2" className="text-gray-600 italic">
            AI accuracy improves with more transaction data. For best results, maintain at least 3 months of transaction history.
          </Typography>
        </CardContent>
      </Card>

      {/* Budget Recommendations Card */}
      <Card className="border-l-4 border-green-500">
        <CardContent className="p-6">
          <Typography variant="h6" className="text-green-700 mb-3 flex items-center">
            <DollarSign className="w-5 h-5 mr-2" />
            Smart Budget Recommendations
          </Typography>
          
          <Typography variant="body1" className="mb-4">
            Our AI analyzes your spending patterns to provide personalized budget recommendations
            that help you achieve your financial goals while maintaining your lifestyle.
          </Typography>
          
          <Box className="mt-4 text-center">
            <Button
              variant="outlined"
              color="primary"
              size="small"
              onClick={() => setActiveTab(4)}
            >
              View Budget Recommendations
            </Button>
          </Box>
        </CardContent>
      </Card>
    </div>
  );

  const renderAnomaliesTab = () => {
    // Anomalies Section
    return insights.anomalies?.length > 0 ? (
      <AnomalyDetection anomalies={insights.anomalies} />
    ) : (
      <Paper elevation={0} className="p-8 text-center">
        <AlertCircle className="text-green-500 w-12 h-12 mx-auto mb-4" />
        <Typography variant="h6" className="mb-2">
          No Anomalies Detected
        </Typography>
        <Typography variant="body1" className="text-gray-600">
          Your spending patterns appear normal. No unusual transactions were detected.
        </Typography>
      </Paper>
    );
  };

  const renderSeasonalTab = () => {
    // Debug what we're getting
    console.log("Rendering seasonal tab with data:", data.seasonal_patterns);
    
    // If no seasonal patterns, show a message
    if (!data.seasonal_patterns || typeof data.seasonal_patterns !== 'object') {
      return (
        <Card className="p-6">
          <CardContent>
            <Typography variant="h6" className="mb-2">Seasonal Analysis</Typography>
            <Typography variant="body1">
              Not enough data available for seasonal analysis. Please ensure you have at least 6 months of transaction data.
            </Typography>
          </CardContent>
        </Card>
      );
    }

    // Safely destructure with defaults for all expected properties
    const { 
      month_spending = {}, 
      quarter_spending = {}, 
      category_seasons = {}, 
      seasonality_strength = 0,
      year_over_year = { growth: {} }
    } = data.seasonal_patterns;

    // Calculate max spending for scaling
    const maxSpending = Math.max(...Object.values(month_spending).map(m => m.mean || 0), 0.01);

    return (
      <div className="space-y-6">
        {/* Overall Seasonality Strength */}
        <Card className="border-l-4 border-purple-500">
          <CardContent className="p-6">
            <Typography variant="h6" className="text-purple-700 mb-3 flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Seasonality Strength
            </Typography>
            <div className="flex items-center justify-between">
              <Typography variant="body1">
                Your spending patterns show {seasonality_strength > 20 ? 'strong' : 'moderate'} seasonality
              </Typography>
              <Chip 
                label={`${(seasonality_strength || 0).toFixed(1)}%`}
                color={seasonality_strength > 20 ? 'error' : 'warning'}
              />
            </div>
            <Box className="mt-4">
              <div className="relative h-4 bg-gray-200 rounded-full">
                <div 
                  className="absolute h-4 bg-purple-500 rounded-full"
                  style={{
                    width: `${Math.min(100, (seasonality_strength / 50) * 100)}%`
                  }}
                />
              </div>
              <Typography variant="caption" className="text-gray-500 mt-1">
                Higher values indicate more pronounced seasonal patterns
              </Typography>
            </Box>
          </CardContent>
        </Card>

        {/* Monthly Spending with Confidence Intervals */}
        {Object.keys(month_spending).length > 0 && (
          <Card className="border-l-4 border-blue-500">
            <CardContent className="p-6">
              <Typography variant="h6" className="text-blue-700 mb-3 flex items-center">
                <Calendar className="w-5 h-5 mr-2" />
                Monthly Spending Patterns
              </Typography>
              <div className="space-y-4">
                {Object.entries(month_spending).map(([month, data]) => (
                  <div key={month} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Typography variant="body1" className="w-24">{month}</Typography>
                      <div className="flex-1 mx-4">
                        <div className="relative h-4 bg-gray-200 rounded-full">
                          <div 
                            className="absolute h-4 bg-blue-500 rounded-full"
                            style={{
                              width: `${Math.min(100, (data.mean / maxSpending) * 100)}%`
                            }}
                          />
                        </div>
                      </div>
                      <div className="text-right w-48">
                        <Typography variant="body2" className="text-gray-600">
                          {formatCurrency(data.mean || 0)} ± {formatCurrency(((data.ci_upper || 0) - (data.ci_lower || 0)) / 2)}
                        </Typography>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Confidence: {data.confidence} transactions</span>
                      <span>Range: {formatCurrency(data.ci_lower)} - {formatCurrency(data.ci_upper)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Quarterly Analysis */}
        {Object.keys(quarter_spending).length > 0 && (
          <Card className="border-l-4 border-green-500">
            <CardContent className="p-6">
              <Typography variant="h6" className="text-green-700 mb-3 flex items-center">
                <LineChart className="w-5 h-5 mr-2" />
                Quarterly Analysis
              </Typography>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(quarter_spending).map(([quarter, data]) => (
                  <div key={quarter} className="space-y-2">
                    <Typography variant="subtitle1" className="font-medium">{quarter}</Typography>
                    <div className="relative h-4 bg-gray-200 rounded-full">
                      <div 
                        className="absolute h-4 bg-green-500 rounded-full"
                        style={{
                          width: `${Math.min(100, (data.mean / maxSpending) * 100)}%`
                        }}
                      />
                    </div>
                    <Typography variant="body2" className="text-gray-600">
                      Avg: {formatCurrency(data.mean)}
                    </Typography>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Peak and Low Months */}
        <Card className="border-l-4 border-orange-500">
          <CardContent className="p-6">
            <Typography variant="h6" className="text-orange-700 mb-3 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              Peak Spending Analysis
            </Typography>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Typography variant="subtitle2" className="text-gray-600">Highest Spending Month</Typography>
                <Typography variant="h5" className="text-red-600">
                  {data.seasonal_patterns.highest_spending_month}
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  {formatCurrency(month_spending[data.seasonal_patterns.highest_spending_month]?.mean || 0)}
                </Typography>
              </div>
              <div>
                <Typography variant="subtitle2" className="text-gray-600">Lowest Spending Month</Typography>
                <Typography variant="h5" className="text-green-600">
                  {data.seasonal_patterns.lowest_spending_month}
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  {formatCurrency(month_spending[data.seasonal_patterns.lowest_spending_month]?.mean || 0)}
                </Typography>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-6xl mx-auto p-4 sm:p-6 lg:p-8">
        <Box className="mb-6">
          <Button
            onClick={handleBack}
            startIcon={<ArrowLeft />}
            variant="outlined"
            className="mb-4"
          >
            Back to Dashboard
          </Button>
          
          <div className="flex flex-col md:flex-row md:justify-between md:items-center">
            <Typography variant="h4" className="mb-2 md:mb-0 flex items-center">
              <BrainCircuit className="w-7 h-7 mr-2 text-blue-600" />
              AI Insights
            </Typography>
            
            <div className="flex items-center gap-2">
              <Chip 
                label="ML Powered" 
                color="primary" 
                variant="outlined"
                size="small"
              />
              {modelsAvailable && (
                <Chip 
                  label="Predictions Available" 
                  color="success" 
                  variant="outlined"
                  size="small"
                />
              )}
            </div>
          </div>
        </Box>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
            <Tab label="Key Insights" id="tab-0" aria-controls="tabpanel-0" />
            {modelsAvailable && (
              <Tab label="Predictions" id="tab-1" aria-controls="tabpanel-1" />
            )}
            <Tab label="Anomalies" id="tab-2" aria-controls="tabpanel-2" />
            <Tab label="Seasonal Patterns" id="tab-3" aria-controls="tabpanel-3" />
            <Tab label="Budget Recommendations" id="tab-4" aria-controls="tabpanel-4" />
            <Tab label="Smart Categorizer" id="tab-5" aria-controls="tabpanel-5" />
          </Tabs>
        </Box>
        
        <div role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
          {activeTab === 0 && renderInsightsTab()}
        </div>
        
        {modelsAvailable && (
          <div role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
            {activeTab === 1 && <PredictionsTab data={data} insights={insights} />}
          </div>
        )}
        
        <div role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
          {activeTab === 2 && renderAnomaliesTab()}
        </div>
        
        <div role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3">
          {activeTab === 3 && renderSeasonalTab()}
        </div>
        
        <div role="tabpanel" hidden={activeTab !== 4} id="tabpanel-4" aria-labelledby="tab-4">
          {activeTab === 4 && <BudgetRecommendations />}
        </div>
        
        <div role="tabpanel" hidden={activeTab !== 5} id="tabpanel-5" aria-labelledby="tab-5">
          {activeTab === 5 && (
            <TransactionCategorizer 
              userTransactions={data.expenses || []} 
              onCategorize={handleCategorizedTransactions}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default AIInsights;