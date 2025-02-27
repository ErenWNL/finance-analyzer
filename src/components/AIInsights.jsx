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
  Divider
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  ArrowLeft,
  BrainCircuit,
} from 'lucide-react';
import PredictionsTab from './PredictionsTab'; // Import the predictions tab component

const AIInsights = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("insights"); // insights, predictions, anomalies, seasonal

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

  // Show loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 p-8 flex justify-center items-center">
        <Card>
          <CardContent className="flex flex-col items-center p-8">
            <CircularProgress className="mb-4" />
            <Typography variant="h6">Loading insights...</Typography>
          </CardContent>
        </Card>
      </div>
    );
  }

  // If data is not available or there's an error
  if (error || !data) {
    return (
      <div className="min-h-screen bg-gray-100 p-8">
        <Card>
          <CardContent>
            <div className="flex flex-col items-center gap-4">
              <Typography variant="h6" color="error">
                {error || "No insights available"}
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

  // Custom tab component
  const CustomTab = ({ id, label, active, onClick, icon }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center px-4 py-2 border-b-2 ${
        active === id
          ? "border-blue-500 text-blue-600"
          : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
      }`}
    >
      {icon}
      <span className="ml-2">{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4">
          <button
            onClick={() => window.history.back()}
            className="bg-[#1A1D24] text-gray-300 hover:text-white px-4 py-2 rounded-lg border border-gray-700 hover:border-blue-500 transition-all duration-200 flex items-center gap-2 hover:bg-[#1E2128]"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to Dashboard
          </button>
        </div>

        <Card className="mt-4">
          <CardContent>
            <div className="flex justify-between items-center mb-4">
              <Typography variant="h5" className="flex items-center">
                AI Insights
                <Chip 
                  label="Powered by ML" 
                  size="small" 
                  color="primary" 
                  variant="outlined"
                  className="ml-2"
                />
                {modelsAvailable && (
                  <Chip 
                    label="ML Models Active" 
                    size="small" 
                    color="success" 
                    variant="outlined"
                    className="ml-2"
                    icon={<BrainCircuit className="h-3 w-3" />}
                  />
                )}
              </Typography>

              <Chip
                label={`AI Confidence: ${transactionCount > 50 ? 'High' : transactionCount > 20 ? 'Medium' : 'Low'}`}
                size="small"
                color={transactionCount > 50 ? "success" : transactionCount > 20 ? "primary" : "warning"}
                variant="outlined"
              />
            </div>

            {/* Custom Tab Navigation */}
            <div className="flex border-b mb-4">
              <CustomTab
                id="insights"
                label="Key Insights"
                active={activeTab}
                onClick={setActiveTab}
                icon={<AlertCircle className="w-4 h-4" />}
              />
              
              {modelsAvailable && (
                <CustomTab
                  id="predictions"
                  label="Predictions"
                  active={activeTab}
                  onClick={setActiveTab}
                  icon={<TrendingUp className="w-4 h-4" />}
                />
              )}
              
              <CustomTab
                id="anomalies"
                label="Anomalies"
                active={activeTab}
                onClick={setActiveTab}
                icon={<AlertCircle className="w-4 h-4" />}
              />
              
              {insights.seasonal_patterns && (
                <CustomTab
                  id="seasonal"
                  label="Seasonal Patterns"
                  active={activeTab}
                  onClick={setActiveTab}
                  icon={<i className="w-4 h-4 far fa-calendar" />}
                />
              )}
            </div>

            {/* Key Insights Tab */}
            {activeTab === "insights" && (
              <div className="space-y-4">
                {/* Spending Insights Section */}
                {insights.spending_insights?.length > 0 && (
                  <div className="p-4 bg-green-50 rounded-lg">
                    <Typography variant="h6" className="text-green-700 mb-3">
                      Spending Insights
                    </Typography>
                    <List>
                      {insights.spending_insights.map((insight, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            {insight.type === 'trend' ? (
                              insight.message.includes('increasing') ? (
                                <TrendingUp className="text-red-500" />
                              ) : (
                                <TrendingDown className="text-green-500" />
                              )
                            ) : insight.type === 'prediction' ? (
                              <TrendingUp className="text-blue-500" />
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
                  </div>
                )}

                {/* Spending Velocity Section */}
                {insights.spending_velocity !== undefined && Math.abs(insights.spending_velocity) > 5 && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <Typography variant="h6" className="text-blue-700 mb-3 flex items-center">
                      {insights.spending_velocity > 0 ? (
                        <TrendingUp className="w-5 h-5 mr-2 text-red-500" />
                      ) : (
                        <TrendingDown className="w-5 h-5 mr-2 text-green-500" />
                      )}
                      Spending Velocity
                    </Typography>
                    <Typography variant="body1">
                      Your spending is {insights.spending_velocity > 0 ? 'accelerating' : 'decelerating'} at a rate of{' '}
                      <span className={insights.spending_velocity > 0 ? 'text-red-600 font-bold' : 'text-green-600 font-bold'}>
                        {Math.abs(insights.spending_velocity).toFixed(1)}%
                      </span>{' '}
                      per month.
                    </Typography>
                  </div>
                )}

                {/* Next Month Prediction (Simple) */}
                {(insights.future_predictions?.length > 0 || insights.next_month_prediction) && (
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <Typography variant="h6" className="text-purple-700 mb-3 flex items-center">
                      <TrendingUp className="w-5 h-5 mr-2" />
                      Next Month's Prediction
                    </Typography>
                    
                    {insights.future_predictions?.length > 0 ? (
                      <div>
                        <Typography variant="body1" className="mb-2">
                          Based on our ML models, your predicted spending for next month is:
                        </Typography>
                        <Chip
                          label={`$${insights.future_predictions[0].total_predicted.toFixed(2)}`}
                          color="primary"
                          className="text-lg"
                        />
                      </div>
                    ) : (
                      <Chip
                        label={`Predicted Spending: $${insights.next_month_prediction?.toFixed(2) || "0.00"}`}
                        color="primary"
                        className="text-lg"
                      />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Predictions Tab */}
            {activeTab === "predictions" && modelsAvailable && (
              <PredictionsTab data={data} insights={insights} />
            )}

            {/* Anomalies Tab */}
            {activeTab === "anomalies" && (
              <div className="space-y-4">
                {/* Anomalies Section */}
                {insights.anomalies?.length > 0 ? (
                  <div className="p-4 bg-red-50 rounded-lg">
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
                            <TrendingUp className="text-red-500" />
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <span className="text-red-700 font-medium">
                                ${anomaly.amount} on {anomaly.category}
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
                  </div>
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
                )}
              </div>
            )}

            {/* Seasonal Patterns Tab */}
            {insights.seasonal_patterns && activeTab === "seasonal" && (
              <div className="p-4 bg-yellow-50 rounded-lg">
                <Typography variant="h6" className="text-yellow-700 mb-3">
                  Seasonal Spending Patterns
                </Typography>
                
                <div className="mb-4">
                  <Typography variant="subtitle1" className="text-gray-700">
                    Highest spending month: <span className="font-bold">{
                      new Date(0, insights.seasonal_patterns.highest_spending_month - 1).toLocaleString('default', { month: 'long' })
                    }</span>
                  </Typography>
                  <Typography variant="subtitle1" className="text-gray-700">
                    Lowest spending month: <span className="font-bold">{
                      new Date(0, insights.seasonal_patterns.lowest_spending_month - 1).toLocaleString('default', { month: 'long' })
                    }</span>
                  </Typography>
                </div>
                
                <Typography variant="body2" className="my-2 text-gray-600 italic">
                  Based on your historical spending patterns across different months
                </Typography>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AIInsights;