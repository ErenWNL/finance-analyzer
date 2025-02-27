import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Calendar
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  Cell
} from 'recharts';

// This component handles the Predictions tab in the AIInsights page
const PredictionsTab = ({ data, insights }) => {
  const futurePredictions = insights?.future_predictions || [];
  const monthlyTotals = data?.monthly_totals || {};

  // Format historical data
  const historicalData = Object.entries(monthlyTotals)
    .map(([month, amount]) => ({
      month,
      amount: parseFloat(amount.toFixed(2)),
      type: 'historical'
    }))
    .sort((a, b) => a.month.localeCompare(b.month))
    .slice(-6); // Last 6 months

  // Format predictions data
  const predictionsData = futurePredictions.map(pred => ({
    month: pred.date,
    amount: parseFloat(pred.total_predicted.toFixed(2)),
    type: 'predicted'
  }));

  // Combine for trend chart
  const combinedData = [...historicalData, ...predictionsData];

  // Format category predictions for the first future month (if available)
  const categoryPredictions = futurePredictions.length > 0 
    ? Object.entries(futurePredictions[0].category_predictions).map(([category, amount]) => ({
        name: category,
        amount: parseFloat(amount.toFixed(2))
      })).sort((a, b) => b.amount - a.amount).slice(0, 5) // Top 5 categories
    : [];

  // Colors for the bar chart
  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c'];

  // Check if there's significant change in spending
  const getChangePercentage = () => {
    if (historicalData.length > 0 && predictionsData.length > 0) {
      const lastHistorical = historicalData[historicalData.length - 1].amount;
      const firstPrediction = predictionsData[0].amount;
      
      if (lastHistorical > 0) {
        return ((firstPrediction - lastHistorical) / lastHistorical) * 100;
      }
    }
    return 0;
  };

  const changePercentage = getChangePercentage();

  return (
    <div className="space-y-6">
      {futurePredictions.length > 0 ? (
        <>
          {/* Spending Trend Chart */}
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                Spending Trend and Predictions
              </Typography>
              
              {changePercentage !== 0 && (
                <Box className="mb-4">
                  <Chip 
                    icon={changePercentage > 0 ? <TrendingUp /> : <TrendingDown />}
                    label={`${changePercentage > 0 ? 'Increase' : 'Decrease'} of ${Math.abs(changePercentage).toFixed(1)}% expected`}
                    color={changePercentage > 10 ? "error" : changePercentage < -5 ? "success" : "primary"}
                    variant="outlined"
                    className="mb-2"
                  />
                </Box>
              )}
              
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <RechartsTooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Amount']} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="amount" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      name="Monthly Spending"
                      dot={(props) => {
                        const { cx, cy, payload } = props;
                        return payload.type === 'predicted' ? (
                          <svg x={cx - 5} y={cy - 5} width={10} height={10} fill="#8884d8" viewBox="0 0 10 10">
                            <circle cx="5" cy="5" r="5" stroke="none" />
                          </svg>
                        ) : (
                          <svg x={cx - 4} y={cy - 4} width={8} height={8} fill="#8884d8" viewBox="0 0 8 8">
                            <rect width="8" height="8" stroke="none" />
                          </svg>
                        );
                      }}
                      strokeDasharray={(props) => {
                        const { dataKey, index, data } = props;
                        return data[index]?.type === 'predicted' ? '5 5' : '0';
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <Typography variant="caption" className="text-gray-500 mt-2 block">
                Dotted line represents predicted future expenses
              </Typography>
            </CardContent>
          </Card>

          {/* Category Predictions */}
          {categoryPredictions.length > 0 && (
            <Card>
              <CardContent>
                <Typography variant="h6" className="mb-4 flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2" />
                  Predicted Category Breakdown (Next Month)
                </Typography>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={categoryPredictions}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <RechartsTooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Predicted Amount']} />
                      <Legend />
                      <Bar dataKey="amount" name="Predicted Amount">
                        {
                          categoryPredictions.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))
                        }
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Predictions Details */}
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">
                Future Spending Predictions
              </Typography>
              <List>
                {futurePredictions.map((prediction, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        <Calendar className="text-blue-500" />
                      </ListItemIcon>
                      <ListItemText
                        primary={<span className="font-medium">{prediction.date}</span>}
                        secondary={
                          <>
                            <div>Predicted total: <strong>${prediction.total_predicted.toFixed(2)}</strong></div>
                            <div className="text-xs mt-1">
                              Top categories: {Object.entries(prediction.category_predictions)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 3)
                                .map(([cat, amount]) => `${cat} ($${amount.toFixed(2)})`)
                                .join(', ')}
                            </div>
                          </>
                        }
                      />
                    </ListItem>
                    {index < futurePredictions.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
          
          {/* Budget Recommendation */}
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">
                AI Budget Recommendation
              </Typography>
              <Paper elevation={1} className="p-4 bg-blue-50">
                <Typography variant="subtitle1" className="font-bold text-blue-700 mb-2">
                  Next Month's Budget Suggestion
                </Typography>
                <Typography variant="body1">
                  Based on your spending patterns and predictions, we recommend a budget of:
                </Typography>
                <Box className="my-3 text-center">
                  <Chip 
                    label={`$${(futurePredictions[0].total_predicted * 0.95).toFixed(2)}`}
                    color="primary"
                    className="text-lg px-4 py-2"
                  />
                </Box>
                <Typography variant="body2" className="text-gray-600 mt-2">
                  This is 5% less than our predicted spending of ${futurePredictions[0].total_predicted.toFixed(2)}.
                </Typography>
              </Paper>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card>
          <CardContent className="text-center py-8">
            <AlertCircle className="text-yellow-500 w-12 h-12 mx-auto mb-4" />
            <Typography variant="h6" className="mb-2">
              Prediction Models Not Available
            </Typography>
            <Typography variant="body1" className="text-gray-600">
              We need more transaction data to generate accurate predictions. 
              Continue adding your expenses, and we'll provide predictions once we have enough data 
              (typically 30+ transactions).
            </Typography>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PredictionsTab;