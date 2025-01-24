// src/components/AIInsights.jsx
import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip
} from '@mui/material';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

const AIInsights = ({ insights }) => {
  if (!insights) return null;

  return (
    <Card className="mb-4">
      <CardContent>
        <Typography variant="h6" className="mb-4">
          AI Insights
        </Typography>

        {insights.anomalies && insights.anomalies.length > 0 && (
          <div className="mb-4">
            <Typography variant="subtitle1" className="mb-2">
              Unusual Spending Patterns
            </Typography>
            <List>
              {insights.anomalies.map((anomaly, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <AlertCircle className="text-red-500" />
                  </ListItemIcon>
                  <ListItemText
                    primary={`${anomaly.category}: $${anomaly.amount}`}
                    secondary={new Date(anomaly.date).toLocaleDateString()}
                  />
                </ListItem>
              ))}
            </List>
          </div>
        )}

        <div className="mb-4">
          <Typography variant="subtitle1" className="mb-2">
            Next Month's Prediction
          </Typography>
          <Chip
            label={`Predicted Spending: $${insights.next_month_prediction?.toFixed(2)}`}
            color="primary"
          />
        </div>

        <div>
          <Typography variant="subtitle1" className="mb-2">
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
                <ListItemText primary={insight.message} />
              </ListItem>
            ))}
          </List>
        </div>
      </CardContent>
    </Card>
  );
};

export default AIInsights;