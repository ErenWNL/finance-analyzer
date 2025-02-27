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
  Box,
  Tooltip
} from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend,
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell 
} from 'recharts';
import { 
  User, 
  LogOut, 
  Menu as MenuIcon, 
  BrainCircuit,
  TrendingUp, 
  TrendingDown, 
  AlertCircle,
  RefreshCw,
  Upload,
  LineChart as LineChartIcon,
  Newspaper
} from 'lucide-react';
import ExpenseManager from './ExpenseManager';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import InfoIcon from '@mui/icons-material/Info';
import { getFirestore, doc, getDoc, updateDoc } from 'firebase/firestore';
import Papa from 'papaparse';
import { v4 as uuidv4 } from 'uuid';

const Dashboard = () => {
  const categories = [
    'Food',
    'Transportation',
    'Housing',
    'Utilities',
    'Entertainment',
    'Healthcare',
    'Shopping',
    'Other'
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

  // State
  const [expenses, setExpenses] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Hooks
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const db = getFirestore();

  // Data cleaning utilities
  const cleanData = {
    date: (value) => {
      if (!value) return null;
      const date = new Date(value);
      return isNaN(date.getTime()) ? null : date.toISOString().split('T')[0];
    },
    
    amount: (value) => {
      if (!value) return null;
      const cleanValue = String(value).replace(/[^0-9.-]/g, '');
      const number = parseFloat(cleanValue);
      return isNaN(number) ? null : Math.abs(number);
    },
    
    category: (value, validCategories) => {
      if (!value) return 'Other';
      const cleaned = String(value).trim();
      const match = validCategories.find(
        cat => cat.toLowerCase() === cleaned.toLowerCase()
      );
      return match || 'Other';
    },
    
    description: (value) => {
      return value ? String(value).trim().slice(0, 200) : '';
    }
  };

  // Validation
  const validateExpenseData = (expense) => {
    if (!expense) return null;
  
    try {
      // Create a new object with validated data
      return {
        ...expense,
        // Ensure date is valid
        date: (() => {
          try {
            const date = new Date(expense.date);
            return isNaN(date.getTime()) ? new Date().toISOString().split('T')[0] : expense.date;
          } catch (e) {
            return new Date().toISOString().split('T')[0];
          }
        })(),
        // Ensure amount is a number
        amount: (() => {
          const amount = parseFloat(expense.amount);
          return isNaN(amount) ? 0 : amount;
        })(),
        // Ensure timestamps are valid
        createdAt: (() => {
          try {
            const date = new Date(expense.createdAt);
            return isNaN(date.getTime()) ? new Date().toISOString() : expense.createdAt;
          } catch (e) {
            return new Date().toISOString();
          }
        })(),
        updatedAt: (() => {
          try {
            const date = new Date(expense.updatedAt);
            return isNaN(date.getTime()) ? new Date().toISOString() : expense.updatedAt;
          } catch (e) {
            return new Date().toISOString();
          }
        })(),
        // Ensure other required fields exist
        category: expense.category || 'Other',
        description: expense.description || '',
        id: expense.id || uuidv4(),
        userId: expense.userId || user.uid
      };
    } catch (e) {
      console.error('Error validating expense:', e);
      return null;
    }
  };

  // Deduplication function
  const deduplicateExpenses = (expenseList) => {
    const seen = new Map();
    return expenseList.filter(expense => {
      if (!expense || !expense.id) return false;
      if (seen.has(expense.id)) return false;
      seen.set(expense.id, true);
      return true;
    });
  };

  // CSV Upload Handler
  const handleCSVUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
  
    setLoading(true);
    const invalidRows = [];
    const validExpenses = [];
  
    try {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: async (results) => {
          results.data.forEach((row, index) => {
            try {
              const cleanedExpense = {
                date: cleanData.date(row.date),
                amount: cleanData.amount(row.amount),
                category: cleanData.category(row.category, categories),
                description: cleanData.description(row.description),
                id: uuidv4(),
                userId: user.uid,
                createdAt: (() => {
                  try {
                    return new Date().toISOString();
                  } catch (e) {
                    console.error('Error creating timestamp:', e);
                    return new Date(0).toISOString(); // fallback to epoch
                  }
                })(),
                updatedAt: (() => {
                  try {
                    return new Date().toISOString();
                  } catch (e) {
                    console.error('Error creating timestamp:', e);
                    return new Date(0).toISOString(); // fallback to epoch
                  }
                })()
              };
  
              const validation = validateExpense(cleanedExpense);
  
              if (validation.valid) {
                validExpenses.push(cleanedExpense);
              } else {
                invalidRows.push({
                  rowNumber: index + 2,
                  error: validation.error,
                  originalData: row
                });
              }
            } catch (err) {
              invalidRows.push({
                rowNumber: index + 2,
                error: 'Row processing error',
                originalData: row
              });
            }
          });
  
          if (validExpenses.length > 0) {
            try {
              const userRef = doc(db, 'users', user.uid);
              const updatedExpenses = deduplicateExpenses([...expenses, ...validExpenses]);
              
              await updateDoc(userRef, {
                expenses: updatedExpenses
              });
  
              setExpenses(updatedExpenses);
              await analyzeExpenses(updatedExpenses);
  
              const successMessage = `Successfully imported ${validExpenses.length} expenses` +
                (invalidRows.length > 0 ? ` (${invalidRows.length} rows skipped)` : '');
              setSuccess(successMessage);
            } catch (error) {
              setError('Error updating database: ' + error.message);
              console.error('Database update error:', error);
            }
          } else {
            setError('No valid transactions found in the CSV file');
          }
        },
        error: (error) => {
          setError('Error processing CSV: ' + error.message);
          console.error('CSV parsing error:', error);
        }
      });
    } catch (err) {
      setError('Error uploading file: ' + err.message);
      console.error('File upload error:', err);
    } finally {
      setLoading(false);
      event.target.value = '';
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

  const handleAIClick = () => {
    handleMenuClose();
    if (analysis) {
      navigate('/ai-insights', { 
        state: { 
          data: analysis  // Pass the entire analysis object, not just ai_insights
        } 
      });
    } else {
      setError('No analysis data available yet. Please add or import some expenses first.');
    }
  };

  const handleFinanceNewsClick = () => {
    navigate('/finance-news');
    handleMenuClose();
  };  

  const handleExpenseUpdate = async (updatedExpenses) => {
    const deduplicatedExpenses = deduplicateExpenses(updatedExpenses);
    setExpenses(deduplicatedExpenses);
    if (deduplicatedExpenses.length > 0) {
      await analyzeExpenses(deduplicatedExpenses);
    }
  };

  // Data loading and analysis
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
        const loadedExpenses = docSnap.data().expenses
          .map(validateExpenseData)
          .filter(expense => expense !== null);
          
        const deduplicatedExpenses = deduplicateExpenses(loadedExpenses);
        setExpenses(deduplicatedExpenses);
        await analyzeExpenses(deduplicatedExpenses);
      }
    } catch (err) {
      console.error('Error loading expenses:', err);
      setError('Failed to load your expenses. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeExpenses = async (expenseData) => {
    try {
      setError(null);
      
      // Validate and clean all expense data before sending to API
      const validatedExpenses = expenseData
        .map(validateExpenseData)
        .filter(expense => expense !== null);
  
      const response = await fetch(`${import.meta.env.VITE_API}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          expenses: validatedExpenses,
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
  

  // Components
  const CSVUploadButton = () => (
    <div className="mb-4">
      <input
        type="file"
        accept=".csv"
        onChange={handleCSVUpload}
        style={{ display: 'none' }}
        id="csv-upload"
      />
      <div className="space-y-2">
        <div className="flex items-center space-x-2">
          <label htmlFor="csv-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<Upload />}
              disabled={loading}
            >
              {loading ? 'Importing...' : 'Import CSV'}
            </Button>
          </label>
        </div>
        <div className="text-sm text-gray-600">
          <p>CSV Format: date (YYYY-MM-DD), amount (number), category, description (optional)</p>
          <p>Valid categories: {categories.join(', ')}</p>
        </div>
      </div>
    </div>
  );


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
        <MenuItem onClick={handleAIClick}>
        <div className="flex items-center space-x-2">
          <BrainCircuit className="w-5 h-5" />
          <span>AI Insights</span>
        </div>
        </MenuItem>
        <MenuItem onClick={handleFinanceNewsClick}>
        <div className="flex items-center space-x-2">
          <Newspaper className="w-5 h-5" />
          <span>Finance News</span>
        </div>
      </MenuItem>
        <MenuItem onClick={handleMenuClose}>Settings</MenuItem>
        <MenuItem onClick={handleLogout}>Logout</MenuItem>
      </Menu>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <Alert severity="error" className="mb-4" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" className="mb-4" onClose={() => setSuccess(null)}>
            {success}
          </Alert>
        )}

        <div className="flex justify-between items-center mb-6">
          <CSVUploadButton />
        </div>

        <ExpenseManager 
          expenses={expenses}
          onUpdate={handleExpenseUpdate}
          categories={categories}
        />

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
                    <RechartsTooltip />
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
                    <RechartsTooltip />
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
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;