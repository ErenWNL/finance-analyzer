import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, CardContent, CircularProgress, Alert, Button,
  AppBar, Toolbar, Typography, IconButton, Menu, MenuItem,
  List, ListItem, ListItemIcon, ListItemText, Chip, Box,
  Tooltip, Select, FormControl, InputLabel, TextField,
  Dialog, DialogTitle, DialogContent, DialogContentText,
  DialogActions, Snackbar, Divider
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar
} from 'recharts';
import { 
  User, LogOut, Menu as MenuIcon, BrainCircuit,
  TrendingUp, TrendingDown, AlertCircle, RefreshCw,
  Upload, LineChart as LineChartIcon, Newspaper,
  Filter, Plus as PlusIcon, Edit2, Trash2, Save,
  Calendar, DollarSign, CreditCard, PieChart as PieChartIcon
} from 'lucide-react';
import ExpenseManager from './ExpenseManager';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import InfoIcon from '@mui/icons-material/Info';
import { getFirestore, doc, getDoc, updateDoc } from 'firebase/firestore';
import Papa from 'papaparse';
import { v4 as uuidv4 } from 'uuid';

// Constants
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];
const CURRENCY_SYMBOL = '₹';
const API_BASE_URL = import.meta.env.VITE_API || 'http://localhost:5000';

const Dashboard = () => {
  // Default categories
  const categories = [
    'Food', 'Transportation', 'Housing', 'Utilities', 
    'Entertainment', 'Healthcare', 'Shopping', 'Other'
  ];

  // Hooks
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const db = getFirestore();

  // State - Data
  const [expenses, setExpenses] = useState([]);
  const [filteredExpenses, setFilteredExpenses] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [edaResults, setEdaResults] = useState(null);
  
  // State - UI
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedMonth, setSelectedMonth] = useState('all');
  const [availableMonths, setAvailableMonths] = useState([]);
  const [clearAllDialogOpen, setClearAllDialogOpen] = useState(false);
  
  // State - Expense Management
  const [activeExpenseId, setActiveExpenseId] = useState(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [editingCardId, setEditingCardId] = useState(null);
  const [editCardAmount, setEditCardAmount] = useState('');
  
  // State - Dashboard view
  const [activeView, setActiveView] = useState('transactions'); // 'transactions', 'analysis', 'eda'

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

  // Extract unique months from expenses
  const extractAvailableMonths = (expenses) => {
    const monthSet = new Set();
    
    expenses.forEach(expense => {
      if (expense.date) {
        const date = new Date(expense.date);
        if (!isNaN(date.getTime())) {
          const monthYear = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
          monthSet.add(monthYear);
        }
      }
    });
    
    return Array.from(monthSet).sort();
  };
  
  // Filter expenses by selected month
  const filterExpensesByMonth = (expenses, selectedMonth) => {
    if (selectedMonth === 'all') {
      return expenses;
    }
    
    return expenses.filter(expense => {
      if (!expense || !expense.date) return false;
      
      const date = new Date(expense.date);
      if (isNaN(date.getTime())) return false;
      
      const expenseMonth = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      return expenseMonth === selectedMonth;
    });
  };
  
  // Update filtered expenses when month selection or expenses change
  useEffect(() => {
    if (expenses.length > 0) {
      const months = extractAvailableMonths(expenses);
      setAvailableMonths(months);
      
      // Apply month filtering
      const filtered = filterExpensesByMonth(expenses, selectedMonth);
      console.log(`Filtering for month ${selectedMonth}: ${filtered.length}/${expenses.length} expenses match`);
      setFilteredExpenses(filtered);
    } else {
      setFilteredExpenses([]);
      setAvailableMonths([]);
    }
  }, [expenses, selectedMonth]);

  // Data loading and analysis
  useEffect(() => {
    if (user?.uid) {
      loadExpenses();
    }
  }, [user]);

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

  // Data loading functions
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
        
        // Load existing analysis if available
        if (docSnap.data().latest_analysis) {
          setAnalysis(docSnap.data().latest_analysis);
        } else {
          await analyzeExpenses(deduplicatedExpenses);
        }
        
        // Perform EDA
        await performExploratoryAnalysis(deduplicatedExpenses);
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
      
      // Skip analysis if no data
      if (!expenseData || expenseData.length === 0) {
        setAnalysis(null);
        return;
      }
      
      // Validate and clean all expense data before sending to API
      const validatedExpenses = expenseData
        .map(validateExpenseData)
        .filter(expense => expense !== null);
      
      // Skip if no valid expenses after filtering
      if (validatedExpenses.length === 0) {
        setAnalysis(null);
        return;
      }
  
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            expenses: validatedExpenses,
            user_id: user.uid
          }),
        });
  
        const data = await response.json();
        
        if (data.status === 'success') {
          setAnalysis(data.data);
        } else {
          console.warn('Analysis returned with error status:', data.message);
          // Don't show error to user, just log it
        }
      } catch (err) {
        console.error('Analysis API error:', err);
        // Don't show error to user, just log it
      } finally {
        setLoading(false);
      }
    } catch (err) {
      console.error('Analysis processing error:', err);
      // Don't show error to user, just log it
    }
  };
  
  const performExploratoryAnalysis = async (expenseData) => {
    try {
      // Skip if no data
      if (!expenseData || expenseData.length === 0) {
        return;
      }
      
      // Validate and clean all expense data before sending to API
      const validatedExpenses = expenseData
        .map(validateExpenseData)
        .filter(expense => expense !== null);
  
      try {
        const response = await fetch(`${API_BASE_URL}/api/exploratory`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            expenses: validatedExpenses
          }),
        });
  
        if (!response.ok) {
          console.warn('EDA endpoint returned error:', response.status);
          return; // Don't set error, just return
        }
  
        const data = await response.json();
        
        if (data.status === 'success') {
          setEdaResults(data.data);
        }
      } catch (err) {
        console.error('EDA error:', err);
        // Just log the error, don't set error state - EDA is not critical
      }
    } catch (err) {
      console.error('EDA processing error:', err);
      // Don't set error state - EDA is not critical
    }
  };

// Event Handlers
const handleMonthChange = (event) => {
  setSelectedMonth(event.target.value);
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
        data: analysis
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

const handleViewChange = (view) => {
  setActiveView(view);
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
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString()
            };

            if (cleanedExpense.date && cleanedExpense.amount && cleanedExpense.category) {
              validExpenses.push(cleanedExpense);
            } else {
              invalidRows.push({
                rowNumber: index + 2,
                error: 'Missing required fields',
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
            await performExploratoryAnalysis(updatedExpenses);

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

// Expense CRUD operations
const handleExpenseAction = (action, expenseId = null) => {
  if (action === 'add') {
    setShowAddDialog(true);
  } else if (action === 'edit') {
    const expenseToEdit = expenses.find(exp => exp.id === expenseId);
    if (expenseToEdit) {
      handleEditStart(expenseToEdit);
    }
  } else if (action === 'delete') {
    setActiveExpenseId(expenseId);
    setShowDeleteDialog(true);
  }
};

const handleEditStart = (expense) => {
  setEditingCardId(expense.id);
  setEditCardAmount(expense.amount.toString());
};

const handleEditSave = async (expenseId) => {
  try {
    if (!editCardAmount || isNaN(parseFloat(editCardAmount))) {
      setError('Please enter a valid amount');
      return;
    }
    
    // Update in the local state first
    const updatedExpenses = expenses.map(expense => 
      expense?.id === expenseId 
        ? { ...expense, amount: parseFloat(editCardAmount), updatedAt: new Date().toISOString() }
        : expense
    );
    
    // Update in Firestore
    const userRef = doc(db, 'users', user.uid);
    await updateDoc(userRef, {
      expenses: updatedExpenses
    });
    
    setExpenses(updatedExpenses);
    const filtered = filterExpensesByMonth(updatedExpenses, selectedMonth);
    setFilteredExpenses(filtered);
    
    if (updatedExpenses.length > 0) {
      await analyzeExpenses(updatedExpenses);
      await performExploratoryAnalysis(updatedExpenses);
    }
    
    setEditingCardId(null);
    setEditCardAmount('');
    setSuccess('Expense updated successfully');
  } catch (err) {
    console.error('Error updating expense:', err);
    setError('Failed to update expense');
  }
};

const handleClearAll = () => {
  setClearAllDialogOpen(true);
};

const confirmClearAll = async () => {
  try {
    setLoading(true);
    const userRef = doc(db, 'users', user.uid);
    
    await updateDoc(userRef, {
      expenses: []
    });
    
    setExpenses([]);
    setFilteredExpenses([]);
    setAnalysis(null);
    setEdaResults(null);
    setClearAllDialogOpen(false);
    setSuccess('All transactions cleared successfully');
  } catch (error) {
    setError('Failed to clear transactions');
    console.error('Clear all error:', error);
  } finally {
    setLoading(false);
  }
};

const handleExpenseUpdate = async (updatedExpenses) => {
  try {
    setLoading(true);
    const deduplicatedExpenses = deduplicateExpenses(updatedExpenses);
    setExpenses(deduplicatedExpenses);
    
    // Recalculate filtered expenses based on the updated expense list
    const filtered = filterExpensesByMonth(deduplicatedExpenses, selectedMonth);
    setFilteredExpenses(filtered);
    
    if (deduplicatedExpenses.length > 0) {
      try {
        await analyzeExpenses(deduplicatedExpenses);
      } catch (analyzeErr) {
        console.error('Error in expense analysis:', analyzeErr);
        // Continue even if analysis fails
      }
      
      try {
        await performExploratoryAnalysis(deduplicatedExpenses);
      } catch (edaErr) {
        console.error('Error in exploratory analysis:', edaErr);
        // Continue even if EDA fails
      }
    } else {
      // Clear analysis data if no expenses
      setAnalysis(null);
      setEdaResults(null);
    }
    
    console.log(`Expenses updated. Total: ${deduplicatedExpenses.length}, Filtered: ${filtered.length}`);
    setSuccess('Expenses updated successfully');
  } catch (err) {
    console.error('Error in handleExpenseUpdate:', err);
    setError('There was an issue updating expenses, but your data has been saved.');
  } finally {
    setLoading(false);
  }
};

// UI Calculations
const filteredTotalAmount = useMemo(() => {
  return filteredExpenses.reduce((sum, exp) => {
    const amount = parseFloat(exp?.amount || 0);
    return sum + (isNaN(amount) ? 0 : amount);
  }, 0);
}, [filteredExpenses]);

const totalAmount = useMemo(() => {
  return expenses.reduce((sum, exp) => {
    const amount = parseFloat(exp?.amount || 0);
    return sum + (isNaN(amount) ? 0 : amount);
  }, 0);
}, [expenses]);

// Formatting Utilities
const formatCurrency = (amount) => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2
  }).format(amount);
};

const formatMonth = (monthStr) => {
  try {
    return new Date(monthStr + '-01').toLocaleString('default', { month: 'long', year: 'numeric' });
  } catch (e) {
    return monthStr;
  }
};

// UI Components
const ClearAllConfirmationDialog = () => (
  <Dialog
    open={clearAllDialogOpen}
    onClose={() => setClearAllDialogOpen(false)}
  >
    <DialogTitle>
      <div className="flex items-center text-red-600">
        <AlertCircle className="w-6 h-6 mr-2" />
        Confirm Delete All
      </div>
    </DialogTitle>
    <DialogContent>
      <div className="space-y-4">
        <DialogContentText>
          Are you sure you want to delete <strong>all</strong> your transactions? This action cannot be undone.
        </DialogContentText>
        <Alert severity="warning" className="mt-3">
          This will permanently remove {expenses.length} transactions from your account.
        </Alert>
      </div>
    </DialogContent>
    <DialogActions>
      <Button 
        onClick={() => setClearAllDialogOpen(false)}
        color="primary"
      >
        Cancel
      </Button>
      <Button 
        onClick={confirmClearAll} 
        color="error" 
        variant="contained"
        startIcon={<Trash2 className="w-4 h-4" />}
      >
        Delete All
      </Button>
    </DialogActions>
  </Dialog>
);

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
        <Tooltip title="Refresh data">
          <span>
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <RefreshCw className={refreshing ? "animate-spin" : ""} size={20} />
            </IconButton>
          </span>
        </Tooltip>
      </div>
      <div className="text-sm text-gray-600">
        <p>CSV Format: date (YYYY-MM-DD), amount (number), category, description (optional)</p>
      </div>
    </div>
  </div>
);

const MonthFilter = () => (
  <FormControl variant="outlined" size="small" style={{ minWidth: 150 }}>
    <InputLabel id="month-filter-label">Filter by Month</InputLabel>
    <Select
      labelId="month-filter-label"
      id="month-filter"
      value={selectedMonth}
      onChange={handleMonthChange}
      label="Filter by Month"
      startAdornment={<Filter size={16} style={{ marginRight: 8 }} />}
    >
      <MenuItem value="all">All Months</MenuItem>
      {availableMonths.map(month => (
        <MenuItem key={month} value={month}>
          {formatMonth(month)}
        </MenuItem>
      ))}
    </Select>
  </FormControl>
);

const StatusSummary = () => {
  // If no expenses exist
  if (expenses.length === 0 && !loading) {
    return (
      <Alert severity="info" className="mt-4 mb-4">
        <div className="flex items-center">
          <AlertCircle className="mr-2" size={20} />
          No expenses found. Add your first expense or import a CSV file to get started.
        </div>
      </Alert>
    );
  }
  
  // If all expenses are being shown
  if (selectedMonth === 'all') {
    return (
      <Chip 
        icon={<InfoIcon />} 
        label={`Showing all ${expenses.length} expenses | Total: ${formatCurrency(totalAmount)}`}
        color="primary" 
        variant="outlined" 
        className="mb-4"
      />
    );
  }
  
  // If filtered expenses exist for a specific month
  if (selectedMonth !== 'all') {
    const monthName = formatMonth(selectedMonth);
    
    return (
      <Chip 
        icon={<InfoIcon />} 
        label={`Showing ${filteredExpenses.length} expenses for ${monthName} | Total: ${formatCurrency(filteredTotalAmount)}`}
        color="primary" 
        variant="outlined" 
        className="mb-4"
      />
    );
  }
  
  return null;
};

const ViewSelector = () => (
  <div className="mb-4 flex space-x-2">
    <Button
      variant={activeView === 'transactions' ? "contained" : "outlined"}
      startIcon={<CreditCard />}
      onClick={() => handleViewChange('transactions')}
    >
      Transactions
    </Button>
    <Button
      variant={activeView === 'analysis' ? "contained" : "outlined"}
      startIcon={<PieChartIcon />}
      onClick={() => handleViewChange('analysis')}
      disabled={!analysis}
    >
      Analysis
    </Button>
    <Button
      variant={activeView === 'eda' ? "contained" : "outlined"}
      startIcon={<BrainCircuit />}
      onClick={() => handleViewChange('eda')}
      disabled={!edaResults}
      >
        Data Insights
      </Button>
    </div>
  );
  
  // Render transaction cards
  const renderTransactionCards = () => {
    if (filteredExpenses.length === 0) {
      return (
        <Alert severity="info" className="my-4">
          No transactions found for the selected period.
        </Alert>
      );
    }

    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {filteredExpenses.map(expense => {
          // Determine category icon based on category name
          let CategoryIcon;
          switch(expense.category) {
            case 'Food':
              CategoryIcon = () => <span role="img" aria-label="food" className="text-xl">🍔</span>;
              break;
            case 'Transportation':
              CategoryIcon = () => <span role="img" aria-label="transportation" className="text-xl">🚗</span>;
              break;
            case 'Housing':
              CategoryIcon = () => <span role="img" aria-label="housing" className="text-xl">🏠</span>;
              break;
            case 'Utilities':
              CategoryIcon = () => <span role="img" aria-label="utilities" className="text-xl">💡</span>;
              break;
            case 'Entertainment':
              CategoryIcon = () => <span role="img" aria-label="entertainment" className="text-xl">🎬</span>;
              break;
            case 'Healthcare':
              CategoryIcon = () => <span role="img" aria-label="healthcare" className="text-xl">⚕️</span>;
              break;
            case 'Shopping':
              CategoryIcon = () => <span role="img" aria-label="shopping" className="text-xl">🛍️</span>;
              break;
            default:
              CategoryIcon = () => <span role="img" aria-label="other" className="text-xl">📌</span>;
          }
          
          // Format date for display
          const expenseDate = new Date(expense.date);
          const formattedDate = expenseDate.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          });
          
          return (
            <Card key={expense.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center">
                    <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center mr-3">
                      <CategoryIcon />
                    </div>
                    <div>
                      <Typography variant="subtitle1" className="font-medium line-clamp-1">
                        {expense.description || expense.category}
                      </Typography>
                      <Typography variant="body2" color="textSecondary" className="text-sm">
                        {formattedDate}
                      </Typography>
                    </div>
                  </div>
                  <Typography variant="h6" className="font-semibold text-right">
                    {editingCardId === expense.id ? (
                      <div className="flex items-center">
                        <TextField
                          type="number"
                          value={editCardAmount}
                          onChange={(e) => setEditCardAmount(e.target.value)}
                          size="small"
                          inputProps={{ step: "0.01", min: "0" }}
                          className="w-24"
                          variant="standard"
                          InputProps={{
                            disableUnderline: true
                          }}
                        />
                        <IconButton 
                          size="small" 
                          color="primary"
                          onClick={() => handleEditSave(expense.id)}
                          className="ml-1"
                        >
                          <Save className="w-4 h-4" />
                        </IconButton>
                      </div>
                    ) : (
                      <>{CURRENCY_SYMBOL} {parseFloat(expense.amount).toFixed(2)}</>
                    )}
                  </Typography>
                </div>
                
                <div className="flex justify-between items-center mt-3 pt-3 border-t border-gray-100">
                  <Chip 
                    label={expense.category} 
                    size="small" 
                    color="primary" 
                    variant="outlined"
                  />
                  <div className="flex space-x-1">
                    <Tooltip title="Edit Amount">
                      <span>
                      <IconButton 
                        size="small" 
                        color="primary"
                        onClick={() => handleExpenseAction('edit', expense.id)}
                      >
                        <Edit2 className="w-4 h-4" />
                      </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <span>
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={() => handleExpenseAction('delete', expense.id)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </IconButton>
                      </span>
                    </Tooltip>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    );
  };
  
  // Render analysis charts and insights
  const renderAnalysisCharts = () => {
    if (!analysis) {
      return (
        <Alert severity="info" className="my-4">
          No analysis data available. Please add some expenses first.
        </Alert>
      );
    }
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        {/* Monthly Spending Trend */}
        <Card>
          <CardContent>
            <div className="flex justify-between items-center mb-4">
              <Typography variant="h6">Monthly Spending Trend</Typography>
              <Tooltip title="Shows your spending patterns over time">
              <span>
                <InfoIcon fontSize="small" color="action" />
                </span>
              </Tooltip>
             
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={Object.entries(analysis.monthly_totals || {}).map(
                  ([month, amount]) => ({
                    month,
                    amount,
                  })
                )}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <RechartsTooltip formatter={(value) => [`${CURRENCY_SYMBOL}${value.toFixed(2)}`, 'Spending']} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="amount"
                  stroke="#8884d8"
                  name="Monthly Spending"
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Expense Categories */}
        <Card>
          <CardContent>
            <div className="flex justify-between items-center mb-4">
              <Typography variant="h6">Expense Categories</Typography>
              <span>
              <Tooltip title="Breakdown of your spending by category">
                <InfoIcon fontSize="small" color="action" />
              </Tooltip>
              </span>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={Object.entries(analysis.category_totals || {}).map(
                    ([category, value]) => ({
                      name: category,
                      value,
                    })
                  )}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name}: ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {Object.entries(analysis.category_totals || {}).map(
                    (entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[index % COLORS.length]}
                      />
                    )
                  )}
                </Pie>
                <RechartsTooltip formatter={(value) => [`${CURRENCY_SYMBOL}${value.toFixed(2)}`, 'Amount']} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI Insights Preview */}
        {analysis.ai_insights && (
          <Card className="md:col-span-2">
            <CardContent>
              <div className="flex justify-between items-center mb-4">
                <Typography variant="h6" className="flex items-center">
                  <BrainCircuit className="w-5 h-5 mr-2" />
                  AI-Powered Insights
                </Typography>
                <Button 
                  variant="outlined" 
                  size="small"
                  onClick={handleAIClick}
                >
                  View Detailed Insights
                </Button>
              </div>
              
              {analysis.ai_insights.spending_insights?.length > 0 ? (
                <List>
                  {analysis.ai_insights.spending_insights.slice(0, 3).map((insight, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {insight.type === 'trend' || insight.type === 'growth_prediction' ? (
                          insight.message.includes('increasing') || insight.message.includes('increase') ? (
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
                  
                  {analysis.ai_insights.spending_insights.length > 3 && (
                    <Box textAlign="center" mt={2}>
                      <Button 
                        size="small" 
                        onClick={handleAIClick}
                      >
                        View All {analysis.ai_insights.spending_insights.length} Insights
                      </Button>
                    </Box>
                  )}
                </List>
              ) : (
                <Alert severity="info">
                  Our AI is analyzing your spending patterns. Add more transactions for better insights.
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Spending Summary */}
        <Card>
          <CardContent>
            <div className="flex justify-between items-center mb-4">
              <Typography variant="h6">Spending Summary</Typography>
              <Tooltip title="Key metrics about your financial activity">
                <span>
                <InfoIcon fontSize="small" color="action" />
                </span>
              </Tooltip>
            </div>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <DollarSign
                    className={analysis.total_spent > 5000 ? "text-red-500" : "text-green-500"}
                  />
                </ListItemIcon>
                <ListItemText
                  primary="Total Spent"
                  secondary={formatCurrency(analysis.total_spent || 0)}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CreditCard />
                </ListItemIcon>
                <ListItemText
                  primary="Average Transaction"
                  secondary={formatCurrency(analysis.average_expense || 0)}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Calendar />
                </ListItemIcon>
                <ListItemText
                  primary="Transaction Count"
                  secondary={analysis.transaction_count || 0}
                />
              </ListItem>
              {analysis.ai_insights?.spending_velocity && (
                <ListItem>
                  <ListItemIcon>
                    {analysis.ai_insights.spending_velocity > 0 ? (
                      <TrendingUp className="text-red-500" />
                    ) : (
                      <TrendingDown className="text-green-500" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary="Spending Velocity"
                    secondary={`${analysis.ai_insights.spending_velocity > 0 ? '+' : ''}${analysis.ai_insights.spending_velocity.toFixed(1)}% month-over-month`}
                  />
                </ListItem>
              )}
            </List>
          </CardContent>
        </Card>
      </div>
    );
  };
  
  // Render exploratory data analysis
  const renderEDA = () => {
    if (!edaResults) {
      return (
        <Alert severity="info" className="my-4">
          <div className="flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <Typography variant="body1">
              Data insights are currently loading or unavailable. Please check back later.
            </Typography>
          </div>
        </Alert>
      );
    }

    // Safely extract basic stats
    const basicStats = edaResults.basic_stats || {};
    
    // Format data for category distribution chart
    const categoryData = edaResults.category_stats 
    ? edaResults.category_stats.map(cat => ({
        name: cat.category || 'Unknown',
        transactions: cat.amount_count || 0,
        spending: cat.amount_sum || 0,
        average: cat.amount_mean || 0
      })) 
    : [];

    
    // Format data for monthly distribution chart
    const monthlyData = [];
    if (edaResults.time_stats && edaResults.time_stats.monthly) {
      Object.entries(edaResults.time_stats.monthly).forEach(([month, data]) => {
        if (data && typeof data === 'object') {
          monthlyData.push({
            month,
            spending: data.sum || 0,
            transactions: data.count || 0
          });
        }
      });
    }
    
    // Format data for day of week distribution
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const dayOfWeekData = [];
    if (edaResults.time_stats && edaResults.time_stats.day_of_week) {
      Object.entries(edaResults.time_stats.day_of_week).forEach(([dayNum, data]) => {
        if (data && typeof data === 'object') {
          const dayIndex = parseInt(dayNum);
          if (!isNaN(dayIndex) && dayIndex >= 0 && dayIndex < 7) {
            dayOfWeekData.push({
              day: dayNames[dayIndex],
              dayNum: dayIndex,
              spending: data.sum || 0,
              transactions: data.count || 0
            });
          }
        }
      });
      // Sort by day of week
      dayOfWeekData.sort((a, b) => a.dayNum - b.dayNum);
    }
    
    return (
      <div className="space-y-6 mt-6">
        {/* Basic Statistics */}
        <Card>
          <CardContent>
            <Typography variant="h6" className="mb-4">Summary Statistics</Typography>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg text-center">
                <Typography variant="h5" className="text-blue-700">{formatCurrency(basicStats.total_spent || 0)}</Typography>
                <Typography variant="body2">Total Spent</Typography>
              </div>
              <div className="p-4 bg-green-50 rounded-lg text-center">
                <Typography variant="h5" className="text-green-700">{formatCurrency(basicStats.avg_transaction || 0)}</Typography>
                <Typography variant="body2">Average Transaction</Typography>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg text-center">
                <Typography variant="h5" className="text-purple-700">{basicStats.total_count || 0}</Typography>
                <Typography variant="body2">Transactions</Typography>
              </div>
              <div className="p-4 bg-yellow-50 rounded-lg text-center">
                <Typography variant="h5" className="text-yellow-700">{formatCurrency(basicStats.median_transaction || 0)}</Typography>
                <Typography variant="body2">Median Transaction</Typography>
              </div>
            </div>
            
            <Divider className="my-4" />
            
            <div className="flex flex-wrap justify-between">
              <div className="p-3">
                <Typography variant="subtitle2">Date Range</Typography>
                <Typography variant="body2">
                  {basicStats.date_range?.start || 'N/A'} to {basicStats.date_range?.end || 'N/A'}
                </Typography>
              </div>
              <div className="p-3">
                <Typography variant="subtitle2">Min Transaction</Typography>
                <Typography variant="body2">{formatCurrency(basicStats.min_transaction || 0)}</Typography>
              </div>
              <div className="p-3">
                <Typography variant="subtitle2">Max Transaction</Typography>
                <Typography variant="body2">{formatCurrency(basicStats.max_transaction || 0)}</Typography>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Category Analysis - Only show if we have data */}
        {categoryData.length > 0 ? (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Category Analysis</Typography>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                    <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                    <Tooltip formatter={(value, name) => [
                      name === 'spending' ? formatCurrency(value) : value,
                      name === 'spending' ? 'Total Spending' : 'Transaction Count'
                    ]} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="spending" name="Total Spending" fill="#8884d8" />
                    <Bar yAxisId="right" dataKey="transactions" name="Transaction Count" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Category Analysis</Typography>
              <Alert severity="info">
                <Typography variant="body2">
                  Category analysis data is not available. Try adding more transactions with different categories.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        )}
        
        {/* Monthly Spending Trends - Only show if we have data */}
        {monthlyData.length > 0 ? (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Monthly Spending Trends</Typography>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={monthlyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value) => [formatCurrency(value), 'Spending']} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="spending" 
                      name="Monthly Spending" 
                      stroke="#8884d8" 
                      activeDot={{ r: 8 }} 
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Monthly Spending Trends</Typography>
              <Alert severity="info">
                <Typography variant="body2">
                  Monthly trend data is not available. Try adding transactions across different months.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        )}
        
        {/* Day of Week Analysis - Only show if we have data */}
        {dayOfWeekData.length > 0 ? (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Day of Week Analysis</Typography>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dayOfWeekData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip formatter={(value) => [formatCurrency(value), 'Spending']} />
                    <Legend />
                    <Bar dataKey="spending" name="Spending by Day of Week" fill="#8884d8">
                      {dayOfWeekData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.dayNum === 0 || entry.dayNum === 6 ? '#ff8042' : '#8884d8'} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <Typography variant="body2" className="mt-4 text-gray-600 italic text-center">
                Weekend days highlighted in orange
              </Typography>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">Day of Week Analysis</Typography>
              <Alert severity="info">
                <Typography variant="body2">
                  Day of week analysis data is not available. Try adding more transactions on different days.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        )}
      </div>
    );
  };

  // Main render
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
            <Tooltip title={user?.email}>
              <span>
              <IconButton color="inherit">
                <User className="w-5 h-5" />
              </IconButton>
              </span>
            </Tooltip>
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
        <MenuItem onClick={handleProfileClick}>
          <ListItemIcon>
            <User size={18} />
          </ListItemIcon>
          <ListItemText>Profile</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleAIClick} disabled={!analysis?.ai_insights}>
          <ListItemIcon>
            <BrainCircuit size={18} />
          </ListItemIcon>
          <ListItemText>AI Insights</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleFinanceNewsClick}>
          <ListItemIcon>
            <Newspaper size={18} />
          </ListItemIcon>
          <ListItemText>Finance News</ListItemText>
        </MenuItem>
      </Menu>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Alerts and Messages */}
        {error && (
          <Alert
            severity="error"
            className="mb-4"
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        )}

        {success && (
          <Alert
            severity="success"
            className="mb-4"
            onClose={() => setSuccess(null)}
          >
            {success}
          </Alert>
        )}

        {/* Top Controls */}
        <div className="flex flex-wrap justify-between items-center mb-6">
          <CSVUploadButton />
          {expenses.length > 0 && <MonthFilter />}
        </div>

        <StatusSummary />

        {expenses.length > 0 && <ViewSelector />}

        {/* Transactions header - only show in transactions view */}
        {activeView === 'transactions' && (
          <div className="flex justify-between items-center mb-4">
            <Typography variant="h6" className="font-medium">
              Transactions
            </Typography>
            <div className="flex space-x-2">
              {expenses.length > 0 && (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Trash2 className="w-4 h-4" />}
                  onClick={handleClearAll}
                  size="small"
                >
                  Clear All
                </Button>
              )}
              <Button
                variant="contained"
                startIcon={<PlusIcon className="w-4 h-4" />}
                onClick={() => handleExpenseAction("add")}
                size="small"
              >
                Add Expense
              </Button>
            </div>
          </div>
        )}

        {/* Loading Indicator */}
        {loading && (
          <Box className="flex justify-center my-8">
            <CircularProgress />
          </Box>
        )}

        {/* Main Content Area */}
        {!loading && (
          <>
            {activeView === 'transactions' && renderTransactionCards()}
            {activeView === 'analysis' && renderAnalysisCharts()}
            {activeView === 'eda' && renderEDA()}
          </>
        )}

        {/* ExpenseManager for handling data operations */}
        <div className="hidden">
          <ExpenseManager
            id="expense-manager"
            key={`expense-manager-${selectedMonth}`}
            expenses={expenses}
            onUpdate={handleExpenseUpdate}
            categories={categories}
            activeExpenseId={activeExpenseId}
            showAddDialog={showAddDialog}
            showDeleteDialog={showDeleteDialog}
            onDialogClose={() => {
              setActiveExpenseId(null);
              setShowAddDialog(false);
              setShowDeleteDialog(false);
            }}
          />
        </div>

        <ClearAllConfirmationDialog />
      </div>
    </div>
  );
};

export default Dashboard;