import React, { useState, useEffect } from 'react';
import { 
  Button, TextField, MenuItem, Dialog, DialogTitle,
  DialogContent, DialogActions, Alert, IconButton,
  Tooltip, DialogContentText, Snackbar,
  Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper, Divider
} from '@mui/material';
import { Plus, Edit2, Trash2, Save, AlertTriangle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { getFirestore, doc, updateDoc, getDoc } from 'firebase/firestore';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import { getAuth } from 'firebase/auth';

const ExpenseManager = ({ 
  expenses, 
  categories, 
  onUpdate, 
  activeExpenseId = null,
  showAddDialog = false,
  showDeleteDialog = false,
  onDialogClose = () => {}
}) => {
  const { user } = useAuth();
  const db = getFirestore();
  const auth = getAuth();
  
  // Dialog states
  const [open, setOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  
  // Status states
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  
  // Editing states
  const [editingId, setEditingId] = useState(null);
  const [editAmount, setEditAmount] = useState('');
  const [expenseToDelete, setExpenseToDelete] = useState(null);
  
  // Form data
  const [newExpense, setNewExpense] = useState({
    date: new Date().toISOString().split('T')[0],
    amount: '',
    category: '',
    description: ''
  });

  // Use the categories from props, or fallback to default if not provided
  const expenseCategories = categories || [
    'Food', 'Transportation', 'Housing', 'Utilities', 
    'Entertainment', 'Healthcare', 'Shopping', 'Other'
  ];

  // Handle dialog close with cleanup
  const handleDialogClose = () => {
    setOpen(false);
    setDeleteConfirmOpen(false);
    setEditingId(null);
    setExpenseToDelete(null);
    onDialogClose();
  };

  // Respond to external triggers
  useEffect(() => {
    if (showAddDialog) {
      setOpen(true);
    }
  }, [showAddDialog]);
  
  useEffect(() => {
    if (activeExpenseId) {
      const expense = expenses.find(exp => exp.id === activeExpenseId);
      if (expense) {
        handleEditStart(expense);
      }
    }
  }, [activeExpenseId, expenses]);
  
  useEffect(() => {
    if (showDeleteDialog && activeExpenseId) {
      const expense = expenses.find(exp => exp.id === activeExpenseId);
      if (expense) {
        handleDelete(expense);
      }
    }
  }, [showDeleteDialog, activeExpenseId, expenses]);

  // Reset form when closing dialog
  useEffect(() => {
    if (!open) {
      setNewExpense({
        date: new Date().toISOString().split('T')[0],
        amount: '',
        category: '',
        description: ''
      });
    }
  }, [open]);

  // Helper Functions
  const deduplicateExpenses = (expenseList) => {
    const seen = new Map();
    return expenseList.filter(expense => {
      if (!expense || !expense.id) return false;
      if (seen.has(expense.id)) return false;
      seen.set(expense.id, true);
      return true;
    });
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(amount);
  };

  // Add a function to auto-categorize a transaction
  const attemptAutoCategorize = async (transaction) => {
    // Only attempt to categorize if the category is Uncategorized and there's a description
    if (transaction.category !== 'Uncategorized' || !transaction.description) {
      return transaction;
    }

    try {
      console.log(`Attempting to auto-categorize transaction: ${transaction.description}`);
      
      // Get auth token
      const currentUser = auth.currentUser;
      if (!currentUser) {
        console.warn("No authenticated user found for auto-categorization");
        return transaction;
      }
      
      const idToken = await currentUser.getIdToken();
      
      // Call the categorization API
      const response = await axios.post('/api/categorizer/categorize', 
        {
          user_id: user.uid,
          description: transaction.description
        },
        {
          headers: {
            'Authorization': `Bearer ${idToken}`
          }
        }
      );
      
      if (response.data.status === 'success' && response.data.data.category) {
        console.log(`Successfully categorized as: ${response.data.data.category}`);
        return {
          ...transaction,
          category: response.data.data.category
        };
      } else {
        console.warn("Categorization API did not return a valid category");
        return transaction;
      }
    } catch (err) {
      console.error('Error auto-categorizing transaction:', err);
      return transaction;
    }
  };

  // CRUD Operations
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
  
    try {
      // Basic validation - category is no longer required
      if (!newExpense.amount || !newExpense.date) {
        setError('Please fill in all required fields');
        return;
      }
  
      // Amount validation
      const parsedAmount = parseFloat(newExpense.amount);
      if (isNaN(parsedAmount) || parsedAmount < 0) {
        setError('Please enter a valid amount');
        return;
      }
  
      // Date validation and formatting
      let formattedDate;
      try {
        const dateObj = new Date(newExpense.date);
        if (isNaN(dateObj.getTime())) {
          throw new Error('Invalid date');
        }
        formattedDate = dateObj.toISOString().split('T')[0];
      } catch (err) {
        setError('Please enter a valid date');
        return;
      }
  
      // Create expense object with validated data
      // If category is empty or 'Other', set it to 'Uncategorized' so the AI model will categorize it
      const expenseToAdd = {
        date: formattedDate,
        amount: parsedAmount,
        category: !newExpense.category || newExpense.category === 'Other' ? 'Uncategorized' : newExpense.category,
        description: newExpense.description || '',
        id: uuidv4(),
        userId: user.uid,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      // Try to auto-categorize if it's uncategorized and has a description
      const categorizedExpense = await attemptAutoCategorize(expenseToAdd);
  
      const updatedExpenses = deduplicateExpenses([...expenses, categorizedExpense]);
  
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });
  
      onUpdate(updatedExpenses);
      
      // Show appropriate success message
      if (categorizedExpense.category !== 'Uncategorized' && expenseToAdd.category === 'Uncategorized') {
        setSuccess(`Expense added and auto-categorized as "${categorizedExpense.category}": ${formatCurrency(parsedAmount)}`);
      } else {
        setSuccess(`Expense added successfully: ${formatCurrency(parsedAmount)}`);
      }
  
      // Reset form and close dialog
      setNewExpense({
        date: new Date().toISOString().split('T')[0],
        amount: '',
        category: '',
        description: ''
      });
      handleDialogClose();
    } catch (err) {
      console.error('Error adding expense:', err);
      setError(err.message || 'Failed to add expense');
    }
  };

  const handleDelete = async (expense) => {
    if (!expense?.id) {
      setError('Invalid expense selected');
      return;
    }
    setExpenseToDelete(expense);
    setDeleteConfirmOpen(true);
  };

  const confirmDelete = async () => {
    try {
      setLoading(true);
      if (!expenseToDelete?.id) {
        setError('No expense selected for deletion');
        handleDialogClose();
        return;
      }
  
      const updatedExpenses = expenses.filter(exp => exp?.id !== expenseToDelete.id);
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });
  
      onUpdate(updatedExpenses);
      handleDialogClose();
      setSuccess('Expense deleted successfully');
    } catch (err) {
      setError('Failed to delete expense');
      console.error('Error deleting expense:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEditStart = (expense) => {
    if (!expense?.id) return;
    setEditingId(expense.id);
    setEditAmount(expense.amount.toString());
  };

  const handleEditSave = async (expenseId) => {
    try {
      setLoading(true);
      if (!editAmount || isNaN(parseFloat(editAmount))) {
        setError('Please enter a valid amount');
        return;
      }

      const updatedExpenses = expenses.map(expense => 
        expense?.id === expenseId 
          ? { 
              ...expense, 
              amount: parseFloat(editAmount),
              updatedAt: new Date().toISOString()
            }
          : expense
      );

      const userRef = doc(db, 'users', user.uid);
      // We need to update the entire expenses array in Firestore, not just the filtered ones
      const docSnap = await getDoc(userRef);
      if (docSnap.exists()) {
        const allExpenses = docSnap.data().expenses || [];
        const updatedAllExpenses = allExpenses.map(expense => 
          expense?.id === expenseId 
            ? { 
                ...expense, 
                amount: parseFloat(editAmount),
                updatedAt: new Date().toISOString()
              }
            : expense
        );
        
        await updateDoc(userRef, {
          expenses: deduplicateExpenses(updatedAllExpenses)
        });
      }

      onUpdate(updatedExpenses);
      setEditingId(null);
      setEditAmount('');
      setSuccess(`Expense updated successfully to ${formatCurrency(parseFloat(editAmount))}`);
    } catch (err) {
      setError('Failed to update expense');
      console.error('Error updating expense:', err);
    } finally {
      setLoading(false);
    }
  };

  // Sort expenses by date (most recent first)
  const sortedExpenses = [...expenses].sort((a, b) => 
    new Date(b.date) - new Date(a.date)
  );

  const RUPEE_SYMBOL = '₹';

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <Button
          variant="contained"
          startIcon={<Plus className="w-4 h-4" />}
          onClick={() => setOpen(true)}
          data-action="add"
        >
          Add Expense
        </Button>
      </div>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} className="mt-4">
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} className="mt-4">
          {success}
        </Alert>
      )}

      {/* Add Expense Dialog */}
      <Dialog 
        open={open} 
        onClose={handleDialogClose}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add New Expense</DialogTitle>
        <form onSubmit={handleSubmit}>
          <DialogContent>
            <div className="space-y-4">
              <TextField
                type="date"
                label="Date"
                value={newExpense.date}
                onChange={(e) => setNewExpense({ ...newExpense, date: e.target.value })}
                fullWidth
                required
                InputLabelProps={{ shrink: true }}
              />
              
              <TextField
                type="number"
                label="Amount"
                value={newExpense.amount}
                onChange={(e) => setNewExpense({ ...newExpense, amount: e.target.value })}
                fullWidth
                required
                InputProps={{
                  startAdornment: <span className="text-gray-500 mr-1">{RUPEE_SYMBOL}</span>,
                }}
                inputProps={{ 
                  step: "0.01",
                  min: "0"
                }}
              />
              
              <TextField
                select
                label="Category"
                value={newExpense.category}
                onChange={(e) => setNewExpense({ ...newExpense, category: e.target.value })}
                fullWidth
                helperText="Leave empty or select 'Other' for auto-categorization"
              >
                <MenuItem value="">-- Auto-categorize --</MenuItem>
                {expenseCategories.map((category) => (
                  <MenuItem key={category} value={category}>
                    {category}
                  </MenuItem>
                ))}
              </TextField>
              
              <TextField
                label="Description"
                value={newExpense.description}
                onChange={(e) => setNewExpense({ ...newExpense, description: e.target.value })}
                fullWidth
                multiline
                rows={2}
              />
            </div>
          </DialogContent>
          
          <DialogActions>
            <Button onClick={handleDialogClose}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              disabled={loading}
            >
              {loading ? 'Adding...' : 'Add'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={handleDialogClose}
      >
        <DialogTitle>
          <div className="flex items-center text-red-600">
            <AlertTriangle className="w-5 h-5 mr-2" />
            Confirm Delete
          </div>
        </DialogTitle>
        <DialogContent>
          <div className="space-y-4">
            <DialogContentText>
              Are you sure you want to delete this expense?
            </DialogContentText>
            {expenseToDelete && (
              <div className="mt-3 p-3 bg-gray-50 rounded space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="font-medium">Amount:</span>
                  <span>{formatCurrency(expenseToDelete.amount)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Category:</span>
                  <span>{expenseToDelete.category}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Date:</span>
                  <span>{new Date(expenseToDelete.date).toLocaleDateString()}</span>
                </div>
                {expenseToDelete.description && (
                  <div className="flex justify-between">
                    <span className="font-medium">Description:</span>
                    <span>{expenseToDelete.description}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={handleDialogClose}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button 
            onClick={confirmDelete} 
            color="error" 
            variant="contained"
            disabled={loading}
          >
            {loading ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Expenses Table */}
      {sortedExpenses.length > 0 ? (
        <TableContainer component={Paper} className="mt-6">
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>Category</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Description</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedExpenses.map((expense) => (
                <TableRow key={expense.id}>
                  <TableCell>
                    {new Date(expense.date).toLocaleDateString()}
                  </TableCell>
                  <TableCell>{expense.category}</TableCell>
                  <TableCell>
                    {editingId === expense.id ? (
                      <div className="flex items-center">
                        <TextField
                          type="number"
                          value={editAmount}
                          onChange={(e) => setEditAmount(e.target.value)}
                          size="small"
                          inputProps={{ 
                            step: "0.01",
                            min: "0"
                          }}
                          className="w-24"
                          variant="standard"
                          InputProps={{
                            startAdornment: <span className="text-gray-500 mr-1">{RUPEE_SYMBOL}</span>,
                          }}
                        />
                      </div>
                    ) : (
                      formatCurrency(expense.amount)
                    )}
                  </TableCell>
                  <TableCell>{expense.description}</TableCell>
                  <TableCell align="right">
                    <div className="flex justify-end space-x-2">
                      {editingId === expense.id ? (
                        <Tooltip title="Save">
                          <IconButton
                            onClick={() => handleEditSave(expense.id)}
                            size="small"
                            color="primary"
                            disabled={loading}
                          >
                            <Save className="w-4 h-4" />
                          </IconButton>
                        </Tooltip>
                      ) : (
                        <Tooltip title="Edit Amount">
                          <IconButton
                            onClick={() => handleEditStart(expense)}
                            size="small"
                            color="primary"
                            data-expense-id={expense.id}
                            data-action="edit"
                            disabled={loading}
                          >
                            <Edit2 className="w-4 h-4" />
                          </IconButton>
                        </Tooltip>
                      )}
                      <Tooltip title="Delete">
                        <IconButton
                          onClick={() => handleDelete(expense)}
                          size="small"
                          color="error"
                          data-expense-id={expense.id}
                          data-action="delete"
                          disabled={loading}
                        >
                          <Trash2 className="w-4 h-4" />
                        </IconButton>
                      </Tooltip>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <Alert severity="info" className="mt-4">
          No expenses found for the selected period.
        </Alert>
      )}
    </div>
  );
};

export default ExpenseManager;