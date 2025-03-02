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
  Tooltip,
  Select,
  FormControl,
  InputLabel,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
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
  Newspaper,
  Filter,
  Plus as PlusIcon,
  Edit2,
  Trash2,
  Save
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

  const RUPEE_SYMBOL = '₹';

  // Hooks
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const db = getFirestore();

  // State
  const [expenses, setExpenses] = useState([]);
  const [filteredExpenses, setFilteredExpenses] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedMonth, setSelectedMonth] = useState('all');
  const [availableMonths, setAvailableMonths] = useState([]);
  const [clearAllDialogOpen, setClearAllDialogOpen] = useState(false);
  
  // Refs for ExpenseManager actions
  const [activeExpenseId, setActiveExpenseId] = useState(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  
  // State for inline editing
  const [editingCardId, setEditingCardId] = useState(null);
  const [editCardAmount, setEditCardAmount] = useState('');

  // Function to handle expense actions from the card UI
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

  // Handle edit directly in the Dashboard
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
          ? { ...expense, amount: parseFloat(editCardAmount) }
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
      setClearAllDialogOpen(false);
      setSuccess('All transactions cleared successfully');
    } catch (error) {
      setError('Failed to clear transactions');
      console.error('Clear all error:', error);
    } finally {
      setLoading(false);
    }
  };

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

  // Handle month selection change
  const handleMonthChange = (event) => {
    setSelectedMonth(event.target.value);
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
  
              // Fix: Proper validation function call
              if (validateExpenseData(cleanedExpense)) {
                validExpenses.push(cleanedExpense);
              } else {
                invalidRows.push({
                  rowNumber: index + 2,
                  error: 'Invalid expense data',
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
    try {
      const deduplicatedExpenses = deduplicateExpenses(updatedExpenses);
      setExpenses(deduplicatedExpenses);
      
      // Recalculate filtered expenses based on the updated expense list
      const filtered = filterExpensesByMonth(deduplicatedExpenses, selectedMonth);
      setFilteredExpenses(filtered);
      
      if (deduplicatedExpenses.length > 0) {
        await analyzeExpenses(deduplicatedExpenses);
      }
      
      console.log(`Expenses updated. Total: ${deduplicatedExpenses.length}, Filtered: ${filtered.length}`);
    } catch (err) {
      console.error('Error in handleExpenseUpdate:', err);
      setError('Failed to update expenses');
    }
  };

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
            {new Date(month + '-01').toLocaleString('default', { month: 'long', year: 'numeric' })}
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
      // Make sure to handle potential NaN values
      const totalAmount = expenses.reduce((sum, exp) => {
        const amount = parseFloat(exp?.amount || 0);
        return sum + (isNaN(amount) ? 0 : amount);
      }, 0);
      
      return (
        <Chip 
          icon={<InfoIcon />} 
          label={`Showing all ${expenses.length} expenses | Total: ${RUPEE_SYMBOL}${totalAmount.toFixed(2)}`}
          color="primary" 
          variant="outlined" 
          className="mb-4"
        />
      );
    }
    
    // If filtered expenses exist for a specific month
    if (selectedMonth !== 'all') {
      // Handle potential NaN values safely
      const totalFiltered = filteredExpenses.reduce((sum, exp) => {
        const amount = parseFloat(exp?.amount || 0);
        return sum + (isNaN(amount) ? 0 : amount);
      }, 0);
      
      const monthName = new Date(selectedMonth + '-01').toLocaleString('default', { month: 'long', year: 'numeric' });
      
      return (
        <Chip 
          icon={<InfoIcon />} 
          label={`Showing ${filteredExpenses.length} expenses for ${monthName} | Total: ${RUPEE_SYMBOL}${totalFiltered.toFixed(2)}`}
          color="primary" 
          variant="outlined" 
          className="mb-4"
        />
      );
    }
    
    return null;
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
            <Tooltip title={user?.email}>
              <IconButton color="inherit">
                <User className="w-5 h-5" />
              </IconButton>
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
        <MenuItem onClick={handleAIClick}>
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
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <AlertCircle size={18} />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
      </Menu>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Only show error messages when there are expenses but analysis failed */}
        {error && expenses.length > 0 && (
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

        <div className="flex flex-wrap justify-between items-center mb-6">
          <CSVUploadButton />
          {expenses.length > 0 && <MonthFilter />}
        </div>

        <StatusSummary />

        {/* Transactions header */}
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

        {/* Transaction Cards Layout - no empty state message here */}
        {filteredExpenses.length > 0 && (
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
                          <>{RUPEE_SYMBOL} {parseFloat(expense.amount).toFixed(2)}</>
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
                          <IconButton 
                            size="small" 
                            color="primary"
                            onClick={() => handleExpenseAction('edit', expense.id)}
                          >
                            <Edit2 className="w-4 h-4" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={() => handleExpenseAction('delete', expense.id)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </IconButton>
                        </Tooltip>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        {loading && (
          <Box className="flex justify-center my-8">
            <CircularProgress />
          </Box>
        )}

        {analysis && !loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <Card>
              <CardContent>
                <div className="flex justify-between items-center mb-4">
                  <Typography variant="h6">Monthly Spending Trend</Typography>
                  <Tooltip title="Shows your spending patterns over time">
                    <InfoIcon fontSize="small" color="action" />
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
                    <RechartsTooltip />
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
                <div className="flex justify-between items-center mb-4">
                  <Typography variant="h6">Expense Categories</Typography>
                  <Tooltip title="Breakdown of your spending by category">
                    <InfoIcon fontSize="small" color="action" />
                  </Tooltip>
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
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <div className="flex justify-between items-center mb-4">
                  <Typography variant="h6">Spending Insights</Typography>
                  <Tooltip title="Key metrics about your financial activity">
                    <InfoIcon fontSize="small" color="action" />
                  </Tooltip>
                </div>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <TrendingUp
                        color={analysis.total_spent > 1000 ? "red" : "green"}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary="Total Spent"
                      secondary={`${RUPEE_SYMBOL}${analysis.total_spent?.toFixed(
                        2
                      )}`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <TrendingDown />
                    </ListItemIcon>
                    <ListItemText
                      primary="Average Expense"
                      secondary={`${RUPEE_SYMBOL}${analysis.average_expense?.toFixed(
                        2
                      )}`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <LineChartIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="Transaction Count"
                      secondary={analysis.transaction_count}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ExpenseManager for handling data operations */}
        <div className="hidden">
          <ExpenseManager
            id="expense-manager"
            key={`expense-manager-${selectedMonth}`}
            expenses={filteredExpenses}
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