import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Alert,
  Button,
  TextField,
  Grid,
  Paper
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  PiggyBank,
  AlertCircle
} from 'lucide-react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { getAuth } from 'firebase/auth';

// Configure axios with base URL
const api = axios.create({
  baseURL: 'http://localhost:5001'
});

// Custom Rupee icon component
const RupeeIcon = () => (
  <span className="w-5 h-5 mr-2 text-gray-500">₹</span>
);

const BudgetRecommendations = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [income, setIncome] = useState('');
  const [savingsGoal, setSavingsGoal] = useState('');

  useEffect(() => {
    console.log('Auth state:', user ? 'Logged in' : 'Not logged in');
    if (!user) {
      setError('Please sign in to access budget recommendations');
    }
  }, [user]);

  const fetchRecommendations = async () => {
    if (!user) {
      setError('Please sign in to access budget recommendations');
      return;
    }

    if (!income) {
      setError('Please enter your monthly income');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get the current Firebase Auth instance and user
      const auth = getAuth();
      const currentUser = auth.currentUser;
      
      if (!currentUser) {
        setError('Authentication error. Please sign in again.');
        navigate('/login');
        return;
      }
      
      const token = await currentUser.getIdToken(true);
      console.log('Current user:', currentUser.email);
      console.log('Fetching recommendations with income:', income);
      
      const response = await api.get(`/api/budget/recommendations?income=${income}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      console.log('Recommendations response:', response.data);
      // Add detailed debugging of response structure
      console.log('Response structure:', {
        total_income: typeof response.data.total_income,
        total_suggested: typeof response.data.total_suggested,
        suggested_savings: typeof response.data.suggested_savings,
        hasChanges: !!response.data.changes,
        hasRecommendations: !!response.data.recommendations
      });
      
      // Ensure all required properties exist with defaults
      const processedData = {
        total_income: response.data.total_income || parseFloat(income),
        total_suggested: response.data.total_suggested || 0,
        suggested_savings: response.data.suggested_savings || 0,
        changes: response.data.changes || {},
        recommendations: response.data.recommendations || {}
      };
      
      setRecommendations(processedData);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      if (err.response?.status === 401) {
        setError('Your session has expired. Please sign in again.');
        navigate('/login');
      } else {
        setError(err.response?.data?.error || 'Failed to fetch recommendations. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    if (!user) {
      setError('Please sign in to train the model');
      return;
    }

    if (!income || !savingsGoal) {
      setError('Please enter both income and savings goal');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get the current Firebase Auth instance and user
      const auth = getAuth();
      const currentUser = auth.currentUser;
      
      if (!currentUser) {
        setError('Authentication error. Please sign in again.');
        navigate('/login');
        return;
      }
      
      const token = await currentUser.getIdToken(true);
      console.log('Current user:', currentUser.email);
      console.log('Training model with income:', income, 'savings goal:', savingsGoal);
      
      const response = await api.post('/api/budget/train', {
        income: parseFloat(income),
        savings_goal: parseFloat(savingsGoal)
      }, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      console.log('Training response:', response.data);
      // Add detailed debugging of response structure from training
      console.log('Training response structure:', {
        success: response.data.message === 'Budget model trained successfully',
        message: response.data.message,
        hasRecommendations: !!response.data.recommendations
      });
      
      if (response.data.message === 'Budget model trained successfully') {
        // Automatically fetch recommendations after training
        await fetchRecommendations();
      }
    } catch (err) {
      console.error('Error training model:', err);
      // Enhanced error logging
      if (err.response) {
        console.error('Error response data:', err.response.data);
        console.error('Error response status:', err.response.status);
        console.error('Error response headers:', err.response.headers);
        
        const errorMsg = err.response.data?.error || err.response.data?.message || `Failed to train model: ${err.message}`;
        setError(errorMsg);
      } else if (err.request) {
        console.error('Error request:', err.request);
        setError('Request was made but no response received. Check your network connection.');
      } else {
        setError(`Failed to train model: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center p-8">
          <Alert severity="warning" className="mb-4">
            Please sign in to access budget recommendations
          </Alert>
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate('/login')}
          >
            Sign In
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center p-8">
          <CircularProgress className="mb-4" />
          <Typography variant="h6">Generating recommendations...</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-4 flex items-center">
            <RupeeIcon />
            Budget Optimization
          </Typography>
          
          <Grid container spacing={2} className="mb-4">
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Monthly Income"
                type="number"
                value={income}
                onChange={(e) => setIncome(e.target.value)}
                InputProps={{
                  startAdornment: <RupeeIcon />
                }}
                error={!!error && !income}
                helperText={!!error && !income ? 'Income is required' : ''}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Monthly Savings Goal"
                type="number"
                value={savingsGoal}
                onChange={(e) => setSavingsGoal(e.target.value)}
                InputProps={{
                  startAdornment: <PiggyBank className="w-5 h-5 mr-2 text-gray-500" />
                }}
                error={!!error && !savingsGoal}
                helperText={!!error && !savingsGoal ? 'Savings goal is required' : ''}
              />
            </Grid>
          </Grid>

          <div className="flex gap-2">
            <Button
              variant="contained"
              color="primary"
              onClick={trainModel}
              disabled={loading || !income || !savingsGoal}
            >
              Train Model
            </Button>
            <Button
              variant="outlined"
              color="primary"
              onClick={fetchRecommendations}
              disabled={loading || !income}
            >
              Get Recommendations
            </Button>
          </div>

          {error && (
            <Alert severity="error" className="mt-4">
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {recommendations && (
        <>
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                Recommended Budget Allocation
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper className="p-4">
                    <Typography variant="subtitle1" className="mb-2">
                      Summary
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Total Income"
                          secondary={`₹${recommendations.total_income ? recommendations.total_income.toFixed(2) : '0.00'}`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Total Suggested Spending"
                          secondary={`₹${recommendations.total_suggested ? recommendations.total_suggested.toFixed(2) : '0.00'}`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Suggested Savings"
                          secondary={`₹${recommendations.suggested_savings ? recommendations.suggested_savings.toFixed(2) : '0.00'}`}
                        />
                      </ListItem>
                    </List>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper className="p-4">
                    <Typography variant="subtitle1" className="mb-2">
                      Category Changes
                    </Typography>
                    {recommendations.changes ? (
                      <List dense>
                        {Object.entries(recommendations.changes)
                          .sort((a, b) => Math.abs(b[1].change) - Math.abs(a[1].change))
                          .map(([category, data]) => (
                            <ListItem key={category}>
                              <ListItemIcon>
                                {data.change > 0 ? (
                                  <TrendingUp className="w-5 h-5 text-green-500" />
                                ) : (
                                  <TrendingDown className="w-5 h-5 text-red-500" />
                                )}
                              </ListItemIcon>
                              <ListItemText
                                primary={category}
                                secondary={
                                  <span className={data.change > 0 ? 'text-green-500' : 'text-red-500'}>
                                    {data.change > 0 ? '+' : ''}
                                    {data.change_percent ? data.change_percent.toFixed(1) : '0.0'}%
                                    ({data.change > 0 ? '+' : ''}₹{data.change ? Math.abs(data.change).toFixed(2) : '0.00'})
                                  </span>
                                }
                              />
                            </ListItem>
                          ))}
                      </List>
                    ) : (
                      <Typography color="text.secondary">No change data available</Typography>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4 flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                Detailed Recommendations
              </Typography>

              {recommendations.recommendations ? (
                <List>
                  {Object.entries(recommendations.recommendations)
                    .sort((a, b) => b[1] - a[1])
                    .map(([category, amount]) => (
                      <React.Fragment key={category}>
                        <ListItem>
                          <ListItemText
                            primary={category}
                            secondary={`₹${amount ? amount.toFixed(2) : '0.00'}`}
                          />
                        </ListItem>
                        <Divider />
                      </React.Fragment>
                    ))}
                </List>
              ) : (
                <Typography color="text.secondary">No detailed recommendations available</Typography>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};

export default BudgetRecommendations; 