import React, { useState, useEffect } from 'react';
import { 
  Button, TextField, MenuItem, Dialog, DialogTitle,
  DialogContent, DialogActions, Alert, IconButton,
  Tooltip, DialogContentText
} from '@mui/material';
import { Plus, Edit2, Trash2, Save } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { getFirestore, doc, setDoc, getDoc, updateDoc } from 'firebase/firestore';

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
    'Shopping',
    'Healthcare',
    'Other'
  ];

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
          setExpenses(userData.expenses);
          onUpdate(userData.expenses);
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
        id: Date.now(),
        userId: user.uid
      };

      const updatedExpenses = [...expenses, expenseToAdd];

      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });

      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);

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
    if (!expense) {
      setError('Invalid expense selected');
      return;
    }
    setExpenseToDelete(expense);
    setDeleteConfirmOpen(true);
  };

  const confirmDelete = async () => {
    try {
      if (!expenseToDelete) {
        setError('No expense selected for deletion');
        setDeleteConfirmOpen(false);
        return;
      }
  
      // Filter out the expense to delete
      const updatedExpenses = expenses.filter(exp => exp && exp.id !== expenseToDelete.id);
      
      // Update Firestore
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });
  
      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);
      setDeleteConfirmOpen(false);
      setExpenseToDelete(null);
      setError(null);
    } catch (err) {
      setError('Failed to delete expense');
      console.error('Error deleting expense:', err);
    }
  };

  const handleEditStart = (expense) => {
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
        expense.id === expenseId 
          ? { ...expense, amount: parseFloat(editAmount) }
          : expense
      );

      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        expenses: updatedExpenses
      });

      setExpenses(updatedExpenses);
      onUpdate(updatedExpenses);
      setEditingId(null);
      setEditAmount('');
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

  return (
    <div>
      <Button
        variant="contained"
        startIcon={<Plus className="w-4 h-4" />}
        onClick={() => setOpen(true)}
        className="mb-4"
      >
        Add Expense
      </Button>

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
            {error && (
              <Alert severity="error" className="mb-4">
                {error}
              </Alert>
            )}

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
    <DialogContentText>
      Are you sure you want to delete this expense?
      {expenseToDelete && (
        <div className="mt-3 p-3 bg-gray-50 rounded">
          <div>Amount: {formatCurrency(expenseToDelete.amount)}</div>
          <div>Category: {expenseToDelete.category}</div>
          <div>Date: {new Date(expenseToDelete.date).toLocaleDateString()}</div>
          {expenseToDelete.description && (
            <div>Description: {expenseToDelete.description}</div>
          )}
        </div>
      )}
    </DialogContentText>
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
      disabled={!expenseToDelete}
    >
      Delete
    </Button>
  </DialogActions>
</Dialog>

      {expenses.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-4">Recent Expenses</h3>
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {expenses.map((expense) => (
                  <tr key={expense.id}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {new Date(expense.date).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {expense.category}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
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
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {expense.description}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex space-x-2">
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
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExpenseManager;