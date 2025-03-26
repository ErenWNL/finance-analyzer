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
  LineChart
} from 'lucide-react';
import PredictionsTab from './PredictionsTab';

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
      setData(location.state.data);
      setLoading(false);
    } else if (location.state?.insights) {
      // Legacy support for old format
      setData({ ai_insights: location.state.insights });
      setLoading(false);
    } else {
      setError("No insights data available");
      setLoading(false);
    }
  }, [location.state]);

  const handleBack = () => {
    navigate('/dashboard');
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
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
                We couldn't find any AI insights for your expenses. Please make sure you have enough transaction data for analysis.
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
    <div className="space-y-4">
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
    </div>
  );

  const renderAnomaliesTab = () => {
    // Anomalies Section
    return insights.anomalies?.length > 0 ? (
      <Card>
        <CardContent className="p-6">
          <Typography variant="h6" className="text-red-700 mb-3 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            Unusual Spending Patterns
          </Typography>
          <Typography variant="body2" className="mb-4 text-gray-700">
            Our AI has detected the following transactions as potentially unusual based on your spending history.
            These were identified using an Isolation Forest machine learning algorithm.
          </Typography>
          <List>
            {insights.anomalies.map((anomaly, index) => (
              <ListItem key={index} className="border-b border-red-100 py-2">
                <ListItemIcon>
                  <AlertCircle className="text-red-500" />
                </ListItemIcon>
                <ListItemText
                  primary={
                    <span className="text-red-700 font-medium">
                      {formatCurrency(anomaly.amount)} on {anomaly.category}
                    </span>
                  }
                  secondary={new Date(anomaly.date).toLocaleDateString()}
                />
              </ListItem>
            ))}
          </List>
          <Box className="mt-4 p-4 bg-gray-50 rounded">
            <Typography variant="subtitle2" className="mb-2">
              Why are these identified as unusual?
            </Typography>
            <Typography variant="body2" className="text-gray-600">
              Transactions may be flagged as unusual if they significantly deviate from your typical spending patterns.
              Factors include amount, category, timing, or frequency compared to your historical spending.
            </Typography>
          </Box>
        </CardContent>
      </Card>
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
    if (!insights.seasonal_patterns) {
      return (
        <Paper elevation={0} className="p-8 text-center">
          <Calendar className="text-blue-500 w-12 h-12 mx-auto mb-4" />
          <Typography variant="h6" className="mb-2">
            Seasonal Analysis Not Available
          </Typography>
          <Typography variant="body1" className="text-gray-600">
            We need at least 12 months of transaction data to perform seasonal analysis. 
            Continue adding expenses to unlock this feature.
          </Typography>
        </Paper>
      );
    }
    
    return (
      <Card>
        <CardContent className="p-6">
          <Typography variant="h6" className="text-yellow-700 mb-4 flex items-center">
            <Calendar className="w-5 h-5 mr-2" />
            Seasonal Spending Patterns
          </Typography>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <Paper className="p-4 bg-green-50 rounded-lg">
              <Typography variant="subtitle1" className="text-green-700 mb-2">
                Highest Spending Month
              </Typography>
              <Typography variant="h5" className="font-bold">
                {new Date(0, insights.seasonal_patterns.highest_spending_month - 1)
                  .toLocaleString('default', { month: 'long' })}
              </Typography>
            </Paper>
            
            <Paper className="p-4 bg-blue-50 rounded-lg">
              <Typography variant="subtitle1" className="text-blue-700 mb-2">
                Lowest Spending Month
              </Typography>
              <Typography variant="h5" className="font-bold">
                {new Date(0, insights.seasonal_patterns.lowest_spending_month - 1)
                  .toLocaleString('default', { month: 'long' })}
              </Typography>
            </Paper>
          </div>
          
          <Typography variant="subtitle1" className="mb-3">
            Monthly Spending Patterns
          </Typography>
          
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {Array.from({length: 12}, (_, i) => i + 1).map(month => {
              const monthName = new Date(0, month - 1).toLocaleString('default', { month: 'short' });
              const amount = insights.seasonal_patterns.month_averages[month] || 0;
              const isHighest = month === insights.seasonal_patterns.highest_spending_month;
              const isLowest = month === insights.seasonal_patterns.lowest_spending_month;
              const bgColor = isHighest ? 'bg-red-100' : isLowest ? 'bg-green-100' : 'bg-gray-50';
              const textColor = isHighest ? 'text-red-700' : isLowest ? 'text-green-700' : '';
              
              return (
                <div key={month} className={`p-2 rounded-lg ${bgColor} text-center`}>
                  <Typography variant="body2" className="font-medium">
                    {monthName}
                  </Typography>
                  <Typography variant="body2" className={`${textColor} text-sm`}>
                    {formatCurrency(amount)}
                  </Typography>
                </div>
              );
            })}
          </div>
          
          <Typography variant="body2" className="mt-6 text-gray-600 italic text-center">
            Based on your historical spending patterns across different months
          </Typography>
        </CardContent>
      </Card>
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
      </div>
    </div>
  );
};

export default AIInsights;