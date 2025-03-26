import React, { useMemo } from 'react';
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
  Divider,
  Alert,
  Grid
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Calendar,
  DollarSign
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
  Cell,
  AreaChart,
  Area
} from 'recharts';

// This component handles the Predictions tab in the AIInsights page
const PredictionsTab = ({ data, insights }) => {
  // Format currency values
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(amount);
  };

  const futurePredictions = insights?.future_predictions || [];
  const monthlyTotals = data?.monthly_totals || {};

  // Format historical data
  const historicalData = useMemo(() => {
    return Object.entries(monthlyTotals)
      .map(([month, amount]) => ({
        month,
        amount: parseFloat(amount.toFixed(2)),
        type: 'historical'
      }))
      .sort((a, b) => a.month.localeCompare(b.month))
      .slice(-6); // Last 6 months
  }, [monthlyTotals]);

  // Format predictions data
  const predictionsData = useMemo(() => {
    return futurePredictions.map(pred => ({
      month: pred.date,
      amount: parseFloat(pred.total_predicted.toFixed(2)),
      type: 'predicted'
    }));
  }, [futurePredictions]);

  // Combine for trend chart
  const combinedData = useMemo(() => {
    return [...historicalData, ...predictionsData];
  }, [historicalData, predictionsData]);

  // Format category predictions for the first future month (if available)
  const categoryPredictions = useMemo(() => {
    if (futurePredictions.length === 0) return [];
    
    return Object.entries(futurePredictions[0].category_predictions)
      .map(([category, amount]) => ({
        name: category,
        amount: parseFloat(amount.toFixed(2))
      }))
      .sort((a, b) => b.amount - a.amount)
      .slice(0, 5); // Top 5 categories
  }, [futurePredictions]);

  // Colors for the bar chart
  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c'];

  // Check if there's significant change in spending
  const changePercentage = useMemo(() => {
    if (historicalData.length > 0 && predictionsData.length > 0) {
      const lastHistorical = historicalData[historicalData.length - 1].amount;
      const firstPrediction = predictionsData[0].amount;
      
      if (lastHistorical > 0) {
        return ((firstPrediction - lastHistorical) / lastHistorical) * 100;
      }
    }
    return 0;
  }, [historicalData, predictionsData]);
  
  // Monthly Budget Recommendation (5% less than prediction)
  const recommendedBudget = useMemo(() => {
    if (futurePredictions.length === 0) return 0;
    return futurePredictions[0].total_predicted * 0.95;
  }, [futurePredictions]);

  if (futurePredictions.length === 0) {
    return (
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
    );
  }

  return (
    <div className="space-y-6">
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
                icon={changePercentage > 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                label={`${changePercentage > 0 ? 'Increase' : 'Decrease'} of ${Math.abs(changePercentage).toFixed(1)}% expected`}
                color={changePercentage > 10 ? "error" : changePercentage < -5 ? "success" : "primary"}
                variant="outlined"
                className="mb-2"
              />
            </Box>
          )}
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={combinedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <RechartsTooltip formatter={(value) => [formatCurrency(value), 'Amount']} />
                <Legend />
                <defs>
                  <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#82ca9d" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="amount" 
                  name="Historical Spending"
                  stroke="#8884d8" 
                  fillOpacity={1}
                  fill="url(#colorHistorical)"
                  activeDot={{ r: 8 }}
                  strokeWidth={2}
                  connectNulls
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return payload.type === 'historical' ? (
                      <svg x={cx - 4} y={cy - 4} width={8} height={8} fill="#8884d8" viewBox="0 0 8 8">
                        <circle cx="4" cy="4" r="4" stroke="none" />
                      </svg>
                    ) : null;
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="amount" 
                  name="Predicted Spending"
                  stroke="#82ca9d" 
                  fillOpacity={1}
                  fill="url(#colorPredicted)"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  connectNulls
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return payload.type === 'predicted' ? (
                      <svg x={cx - 4} y={cy - 4} width={8} height={8} fill="#82ca9d" viewBox="0 0 8 8">
                        <circle cx="4" cy="4" r="4" stroke="none" />
                      </svg>
                    ) : null;
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <Typography variant="caption" className="text-gray-500 mt-2 block">
            Dashed line represents predicted future expenses
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
                <BarChart data={categoryPredictions} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 'dataMax']} />
                  <YAxis type="category" dataKey="name" width={100} />
                  <RechartsTooltip formatter={(value) => [formatCurrency(value), 'Predicted Amount']} />
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

      {/* Budget Recommendation */}
      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-4 flex items-center">
            <DollarSign className="w-5 h-5 mr-2" />
            AI Budget Recommendation
          </Typography>
          <Paper elevation={1} className="p-6 bg-blue-50 rounded-lg">
            <Typography variant="subtitle1" className="font-bold text-blue-700 mb-3">
              Next Month's Budget Suggestion
            </Typography>
            <Typography variant="body1" className="mb-4">
              Based on your spending patterns and predictions, we recommend a budget of:
            </Typography>
            <Box className="my-4 text-center">
              <Chip 
                label={formatCurrency(recommendedBudget)}
                color="primary"
                className="text-lg px-4 py-4"
              />
            </Box>
            <Typography variant="body2" className="text-gray-600 mb-4">
              This recommendation is 5% less than our predicted spending of {formatCurrency(futurePredictions[0].total_predicted)}.
              Adopting this budget can help you save {formatCurrency(futurePredictions[0].total_predicted * 0.05)} next month.
            </Typography>
            
            <Alert severity="info" className="mt-3">
              <Typography variant="body2">
                <strong>Budget Tips:</strong> Focus on reducing spending in your highest categories first. 
                Even small reductions in large expense categories can have a significant impact on your overall budget.
              </Typography>
            </Alert>
          </Paper>
        </CardContent>
      </Card>
          
      {/* Predictions Details */}
      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-4 flex items-center">
            <Calendar className="w-5 h-5 mr-2" />
            Future Spending Predictions
          </Typography>
          <Grid container spacing={2}>
            {futurePredictions.map((prediction, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Paper elevation={2} className="p-4 h-full">
                  <Typography variant="subtitle1" className="font-bold mb-2 flex items-center">
                    <Calendar className="w-4 h-4 mr-2 text-blue-500" />
                    {prediction.date}
                  </Typography>
                  <Typography variant="h6" className="mb-3 text-indigo-600">
                    {formatCurrency(prediction.total_predicted)}
                  </Typography>
                  
                  <Divider className="my-2" />
                  
                  <Typography variant="caption" className="text-gray-500">
                    Top Categories:
                  </Typography>
                  <List dense disablePadding>
                    {Object.entries(prediction.category_predictions)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([category, amount], catIndex) => (
                        <ListItem key={catIndex} disablePadding className="py-1">
                          <ListItemIcon className="min-w-0 mr-1">
                            <span 
                              className="w-2 h-2 rounded-full inline-block" 
                              style={{ backgroundColor: COLORS[catIndex % COLORS.length] }} 
                            />
                          </ListItemIcon>
                          <ListItemText 
                            primary={`${category}: ${formatCurrency(amount)}`}
                            primaryTypographyProps={{ variant: 'body2' }}
                          />
                        </ListItem>
                      ))
                    }
                  </List>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
      
      {/* Prediction Explanation */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" className="font-medium mb-2">
            How Our Predictions Work
          </Typography>
          <Typography variant="body2" className="text-gray-600 mb-3">
            Our AI uses machine learning algorithms to analyze your spending patterns and predict future expenses.
            Specifically, we use:
          </Typography>
          <Box component="ul" className="list-disc pl-5 space-y-1 text-gray-600 text-sm">
            <li>Random Forest regression for overall spending predictions</li>
            <li>Linear regression for category-specific predictions</li>
            <li>Time-based features including day of month, month, and seasonal patterns</li>
            <li>Category distribution analysis from your recent spending habits</li>
          </Box>
          <Alert severity="info" className="mt-4">
            <Typography variant="body2">
              Predictions become more accurate as you add more transactions. For best results, maintain a consistent
              record of your expenses.
            </Typography>
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictionsTab;