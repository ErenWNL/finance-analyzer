import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box
} from '@mui/material';
import { AlertCircle } from 'lucide-react';

const AnomalyDetection = ({ anomalies }) => {
  // Format currency values
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(amount);
  };

  // Format date with fallback
  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown date';
    
    // Try parsing the date
    const date = new Date(dateStr);
    
    // Check if date is valid
    if (isNaN(date.getTime())) {
      // Try alternative formats if standard parsing fails
      if (typeof dateStr === 'string') {
        // Try ISO format with different separators
        const matches = dateStr.match(/(\d{4})[/-](\d{1,2})[/-](\d{1,2})/);
        if (matches) {
          const [_, year, month, day] = matches;
          return new Date(year, month - 1, day).toLocaleDateString();
        }
      }
      return 'Date unavailable';
    }
    
    return date.toLocaleDateString();
  };

  return (
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
          {anomalies.map((anomaly, index) => (
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
                secondary={
                  <span>
                    {formatDate(anomaly.date)}
                    {anomaly.anomaly_reason && (
                      <span className="block text-xs text-gray-600 mt-1">
                        Reason: {anomaly.anomaly_reason}
                      </span>
                    )}
                  </span>
                }
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
  );
};

export default AnomalyDetection; 