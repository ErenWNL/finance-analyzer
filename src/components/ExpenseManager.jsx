import React, { useState, useEffect } from 'react';
import { 
  Button, TextField, MenuItem, Dialog, DialogTitle,
  DialogContent, DialogActions, Alert, IconButton,
  Tooltip, DialogContentText,
  Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper
} from '@mui/material';
import { Plus, Edit2, Trash2, Save } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { getFirestore, doc, setDoc, getDoc, updateDoc } from 'firebase/firestore';
import { v4 as uuidv4 } from 'uuid';

const ExpenseManager = ({ onUpdate }) => {
  const { user } = useAuth();
  const db = getFirestore();
  
  const [expenses, setExpenses] = useState([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [editAmount, setEditAmount] = useState('');
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [expenseToDelete, setExpenseToDelete] = useState(null);
  const [success, setSuccess] = useState(null);
  
  const [newExpense, setNewExpense] = useState({
    date: new Date().toISOString().split('T')[0],
    amount: '',
    category: '',
    description: ''
  });

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

  useEffect(() => {
    if (user?.uid) {
      loadExpenses();
    }
  }, [user]);

  const loadExpenses = async () => {
    try {
      const userRef = doc(db, 'users', user.uid);
      const docSnap = await getDoc(userRef);

      if (docSnap.exists()) {
        const userData = docSnap.data();
        if (userData.expenses) {
          const processedExpenses = userData.expenses.map(expense => ({
            ...expense,
            id: expense.id || uuidv4()
          }));
          const uniqueExpenses = deduplicateExpenses(processedExpenses);
          setExpenses(uniqueExpenses);
          onUpdate(uniqueExpenses);
        }
      } else {
        await setDoc(userRef, { expenses: [] });
      }
    } catch (err) {
      console.error('Error loading expenses:', err);
      setError('Failed to load expenses');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      if (!newExpense.amount || !newExpense.category) {
        setError('Please fill in all required fields');
        return;
      }

      const expenseToAdd = {
        ...newExpense,
        amount: parseFloat(newExpense.amount),
        date: new Date(newExpense.date).toISOString().split('T')[0],
        id: uuidv4(),
        userId: user.uid,
        createdAt: new Date().toISOString()
      };

      const updatedExpenses = deduplicateExpenses([...expenses, expenseToAdd]);

      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });

      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);
      setSuccess('Expense added successfully');

      setNewExpense({
        date: new Date().toISOString().split('T')[0],
        amount: '',
        category: '',
        description: ''
      });
      setOpen(false);
    } catch (err) {
      setError(err.message);
      console.error('Error adding expense:', err);
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
      if (!expenseToDelete?.id) {
        setError('No expense selected for deletion');
        setDeleteConfirmOpen(false);
        return;
      }
  
      const updatedExpenses = expenses.filter(exp => exp?.id !== expenseToDelete.id);
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });
  
      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);
      setDeleteConfirmOpen(false);
      setExpenseToDelete(null);
      setSuccess('Expense deleted successfully');
    } catch (err) {
      setError('Failed to delete expense');
      console.error('Error deleting expense:', err);
    }
  };

  const handleEditStart = (expense) => {
    if (!expense?.id) return;
    setEditingId(expense.id);
    setEditAmount(expense.amount.toString());
  };

  const handleEditSave = async (expenseId) => {
    try {
      if (!editAmount || isNaN(parseFloat(editAmount))) {
        setError('Please enter a valid amount');
        return;
      }

      const updatedExpenses = expenses.map(expense => 
        expense?.id === expenseId 
          ? { ...expense, amount: parseFloat(editAmount) }
          : expense
      );

      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: deduplicateExpenses(updatedExpenses)
      });

      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);
      setEditingId(null);
      setEditAmount('');
      setSuccess('Expense updated successfully');
    } catch (err) {
      setError('Failed to update expense');
      console.error('Error updating expense:', err);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  // Sort expenses by date (most recent first)
  const sortedExpenses = [...expenses].sort((a, b) => 
    new Date(b.date) - new Date(a.date)
  );

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <Button
          variant="contained"
          startIcon={<Plus className="w-4 h-4" />}
          onClick={() => setOpen(true)}
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
        onClose={() => setOpen(false)}
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
                required
              >
                {categories.map((category) => (
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
            <Button onClick={() => setOpen(false)}>Cancel</Button>
            <Button type="submit" variant="contained">Add</Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={() => {
          setDeleteConfirmOpen(false);
          setExpenseToDelete(null);
        }}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
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
            onClick={() => {
              setDeleteConfirmOpen(false);
              setExpenseToDelete(null);
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={confirmDelete} 
            color="error" 
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Expenses Table */}
      {sortedExpenses.length > 0 && (
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
                      />
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
      )}
    </div>
  );
};

export default ExpenseManager;