import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CircularProgress, 
  Alert, 
  Button,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Box
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';
import { 
  User, 
  LogOut, 
  Menu as MenuIcon, 
  TrendingUp, 
  TrendingDown, 
  AlertCircle,
  RefreshCw
} from 'lucide-react';
import ExpenseManager from './ExpenseManager';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import InfoIcon from '@mui/icons-material/Info';
import { getFirestore, doc, getDoc } from 'firebase/firestore';

const Dashboard = () => {
  const [expenses, setExpenses] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const db = getFirestore();

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  useEffect(() => {
    if (user?.uid) {
      loadExpenses();
    }
  }, [user]);

  const loadExpenses = async () => {
    try {
      setLoading(true);
      setError(null);
      const userRef = doc(db, 'users', user.uid);
      const docSnap = await getDoc(userRef);

      if (docSnap.exists() && docSnap.data().expenses) {
        const loadedExpenses = docSnap.data().expenses;
        setExpenses(loadedExpenses);
        await analyzeExpenses(loadedExpenses);
      }
    } catch (err) {
      console.error('Error loading expenses:', err);
      setError('Failed to load your expenses. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadExpenses().finally(() => setRefreshing(false));
  };

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await logout();
      setExpenses([]);
      setAnalysis(null);
      navigate('/login');
    } catch (error) {
      setError('Failed to log out. Please try again.');
    }
  };

  const handleProfileClick = () => {
    handleMenuClose();
    navigate('/profile');
  };

  const analyzeExpenses = async (expenseData) => {
    try {
      setError(null);
      
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          expenses: expenseData,
          user_id: user.uid
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze expenses');
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setAnalysis(data.data);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to analyze expenses. Please try again.');
      return null;
    }
  };

  const handleExpenseUpdate = (updatedExpenses) => {
    setExpenses(updatedExpenses);
    if (updatedExpenses.length > 0) {
      analyzeExpenses(updatedExpenses);
    }
  };

  const AIInsightsCard = ({ insights }) => {
    if (!insights) return null;
  
    return (
      <Card className="col-span-2">
        <CardContent>
          <Box className="flex justify-between items-center mb-6">
            <Typography variant="h6" className="flex items-center">
              <span className="mr-2">AI Insights</span>
              <Chip 
                label="Powered by ML" 
                size="small" 
                color="primary" 
                variant="outlined" 
              />
            </Typography>
            <IconButton 
              onClick={handleRefresh} 
              disabled={refreshing}
              size="small"
              className="hover:bg-gray-100"
            >
              <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
            </IconButton>
          </Box>
  
          {/* Anomaly Detection Section */}
          {insights.anomalies?.length > 0 && (
            <div className="mb-8 p-4 bg-red-50 rounded-lg">
              <Typography variant="subtitle1" className="flex items-center text-red-700 mb-3">
                <AlertCircle className="w-5 h-5 mr-2" />
                Unusual Spending Patterns Detected
              </Typography>
              <List>
                {insights.anomalies.map((anomaly, index) => (
                  <ListItem key={index} className="pl-0">
                    <ListItemIcon>
                      <TrendingUp className="text-red-500 w-5 h-5" />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <span className="text-red-700 font-medium">
                          ${anomaly.amount} on {anomaly.category}
                        </span>
                      }
                      secondary={`on ${new Date(anomaly.date).toLocaleDateString()}`}
                    />
                  </ListItem>
                ))}
              </List>
            </div>
          )}
  
          {/* Prediction Section */}
          {insights.next_month_prediction && (
            <div className="mb-8 p-4 bg-blue-50 rounded-lg">
              <Typography variant="subtitle1" className="flex items-center text-blue-700 mb-3">
                <TrendingUp className="w-5 h-5 mr-2" />
                Next Month's Prediction
              </Typography>
              <div className="flex items-center space-x-2">
                <Chip
                  label={`Predicted Spending: $${insights.next_month_prediction.toFixed(2)}`}
                  color="primary"
                  variant="outlined"
                  className="text-lg"
                />
                <Tooltip title="Based on your spending patterns">
                  <InfoIcon className="w-4 h-4 text-blue-500 cursor-help" />
                </Tooltip>
              </div>
            </div>
          )}
  
          {/* Spending Insights Section */}
          {insights.spending_insights?.length > 0 && (
            <div className="p-4 bg-green-50 rounded-lg">
              <Typography variant="subtitle1" className="flex items-center text-green-700 mb-3">
                <LineChart className="w-5 h-5 mr-2" />
                Smart Spending Analysis
              </Typography>
              <List>
                {insights.spending_insights.map((insight, index) => (
                  <ListItem key={index} className="pl-0">
                    <ListItemIcon>
                      {insight.type === 'trend' ? (
                        insight.message.includes('increasing') ? (
                          <TrendingUp className="text-red-500 w-5 h-5" />
                        ) : (
                          <TrendingDown className="text-green-500 w-5 h-5" />
                        )
                      ) : (
                        <AlertCircle className="text-blue-500 w-5 h-5" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={insight.message}
                      className={`${
                        insight.type === 'trend' && insight.message.includes('increasing')
                          ? 'text-red-600'
                          : 'text-green-600'
                      }`}
                    />
                  </ListItem>
                ))}
              </List>
            </div>
          )}
  
          {/* AI Model Confidence */}
          <div className="mt-4 flex justify-end">
            <Chip
              label={`AI Confidence: High`}
              size="small"
              color="success"
              variant="outlined"
            />
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <AppBar position="static">
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={handleMenuOpen}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" className="flex-grow">
            Finance Analyzer
          </Typography>

          <div className="flex items-center gap-2">
            <User className="w-5 h-5" />
            <Typography variant="body1">
              {user?.email}
            </Typography>
            <Button color="inherit" onClick={handleLogout}>
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </Toolbar>
      </AppBar>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleProfileClick}>Profile</MenuItem>
        <MenuItem onClick={handleMenuClose}>Settings</MenuItem>
        <MenuItem onClick={handleLogout}>Logout</MenuItem>
      </Menu>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <Alert severity="error" className="mb-4" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <ExpenseManager onUpdate={handleExpenseUpdate} />

        {loading && (
          <Box className="flex justify-center my-8">
            <CircularProgress />
          </Box>
        )}

        {analysis && !loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <Card>
              <CardContent>
                <Typography variant="h6" className="mb-4">
                  Monthly Spending Trend
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={Object.entries(analysis.monthly_totals || {}).map(([month, amount]) => ({
                    month,
                    amount
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="amount" 
                      stroke="#8884d8" 
                      name="Spending"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" className="mb-4">
                  Expense Categories
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(analysis.category_totals || {}).map(([category, value]) => ({
                        name: category,
                        value
                      }))}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label
                    >
                      {Object.entries(analysis.category_totals || {}).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" className="mb-4">
                  Summary
                </Typography>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Typography>Total Spent:</Typography>
                    <Typography className="font-medium">
                      ${analysis.total_spent?.toFixed(2)}
                    </Typography>
                  </div>
                  <div className="flex justify-between">
                    <Typography>Average Expense:</Typography>
                    <Typography className="font-medium">
                      ${analysis.average_expense?.toFixed(2)}
                    </Typography>
                  </div>
                  <div className="flex justify-between">
                    <Typography>Transaction Count:</Typography>
                    <Typography className="font-medium">
                      {analysis.transaction_count}
                    </Typography>
                  </div>
                </div>
              </CardContent>
            </Card>

            {analysis.ai_insights && (
              <AIInsightsCard insights={analysis.ai_insights} />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;