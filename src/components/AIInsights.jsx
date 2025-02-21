import React from 'react';
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
  Button
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  ArrowLeft
} from 'lucide-react';

const AIInsights = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const insights = location.state?.insights;

  const handleBack = () => {
    navigate('/dashboard');
  };

  // If no insights data is available
  if (!insights) {
    return (
      <div className="min-h-screen bg-gray-100 p-8">
        <Card>
          <CardContent>
            <div className="flex flex-col items-center gap-4">
              <Typography variant="h6" color="error">
                No insights available
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

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
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

        <Card>
          <CardContent>
            <Typography variant="h5" className="mb-6">
              AI Insights
              <Chip 
                label="Powered by ML" 
                size="small" 
                color="primary" 
                variant="outlined"
                className="ml-2"
              />
            </Typography>

            {/* Anomalies Section */}
            {insights.anomalies?.length > 0 && (
              <div className="mb-6 p-4 bg-red-50 rounded-lg">
                <Typography variant="h6" className="text-red-700 mb-3 flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  Unusual Spending Patterns
                </Typography>
                <List>
                  {insights.anomalies.map((anomaly, index) => (
                    <ListItem key={index}>
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
              </div>
            )}

            {/* Prediction Section */}
            {insights.next_month_prediction && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <Typography variant="h6" className="text-blue-700 mb-3 flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2" />
                  Next Month's Prediction
                </Typography>
                <Chip
                  label={`Predicted Spending: $${insights.next_month_prediction.toFixed(2)}`}
                  color="primary"
                  className="text-lg"
                />
              </div>
            )}

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
                        ) : (
                          <AlertCircle className="text-blue-500" />
                        )}
                      </ListItemIcon>
                      <ListItemText 
                        primary={insight.message}
                        className={
                          insight.type === 'trend' && insight.message.includes('increasing')
                            ? 'text-red-600'
                            : 'text-green-600'
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </div>
            )}

            {/* AI Confidence */}
            <div className="mt-4 flex justify-end">
              <Chip
                label="AI Confidence: High"
                size="small"
                color="success"
                variant="outlined"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AIInsights;