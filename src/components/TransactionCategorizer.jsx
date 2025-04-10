import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Chip,
  Alert,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  BrainCircuit,
  AlertCircle,
  Info,
  Check,
  Upload,
  Loader,
  PenLine,
  X
} from 'lucide-react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { getAuth } from 'firebase/auth';

const TransactionCategorizer = ({ userTransactions, onCategorize }) => {
  const { user } = useAuth();
  const auth = getAuth();
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [modelStats, setModelStats] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [sampleText, setSampleText] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showTrainingDialog, setShowTrainingDialog] = useState(false);
  const [categorizedTransactions, setCategorizedTransactions] = useState([]);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  
  const categories = [
    'Food & Dining', 'Shopping', 'Housing', 'Transportation', 
    'Healthcare', 'Entertainment', 'Utilities', 'Travel', 
    'Education', 'Income', 'Other', 'Groceries', 'Insurance',
    'Personal Care', 'Gifts & Donations', 'Investments',
    'Subscriptions', 'Taxes', 'Business Expenses', 'Fees & Charges'
  ];

  // Add detailed logging on component mount to debug auth issues
  useEffect(() => {
    console.log("TransactionCategorizer mounted - Auth context details:");
    console.log("user exists:", !!user);
    if (user) {
      console.log("user.uid:", user.uid);
      console.log("user type:", typeof user);
      console.log("user keys:", Object.keys(user));
    } else {
      console.log("user is null or undefined");
      
      // Get auth object from localStorage for debugging
      try {
        const localAuth = localStorage.getItem('authUser');
        if (localAuth) {
          console.log("Found auth data in localStorage:", localAuth.substring(0, 100) + "...");
        } else {
          console.log("No auth data in localStorage");
        }
      } catch (e) {
        console.log("Error checking localStorage:", e);
      }
    }
  }, []);

  // First check if user is authenticated
  useEffect(() => {
    console.log("Auth check - Current user state:", user ? `User ID: ${user.uid}` : "No user");
    
    if (user && user.uid) {
      console.log("User is authenticated with ID:", user.uid);
      setIsDemoMode(false);
      setAuthChecked(true);
    } else {
      console.log("No authenticated user or user ID is missing");
      setIsDemoMode(true);
      setAuthChecked(true);
    }
  }, [user]);

  // Then check if model is trained
  useEffect(() => {
    const checkModel = async () => {
      if (!authChecked) return;
      
      try {
        setLoading(true);
        
        if (!user) {
          // No user, so definitely demo mode
          setModelStats({
            is_trained: false,
            is_demo: true,
            categories: categories
          });
          setLoading(false);
          return;
        }
        
        // For logged in users, check if model exists
        console.log("Checking model for user:", user.uid);
        
        // Get authentication token from Firebase Auth
        const currentUser = auth.currentUser;
        if (!currentUser) {
          throw new Error("No authenticated user found");
        }
        
        const idToken = await currentUser.getIdToken();
        
        const response = await axios.post('/api/categorizer/categorize', 
          {
            user_id: user.uid,
            description: 'TEST TRANSACTION'
          },
          {
            headers: {
              'Authorization': `Bearer ${idToken}`
            }
          }
        );
        
        console.log("Model status response:", response.data);
        
        setModelStats({
          is_trained: response.data.data.is_trained,
          categories: categories
        });
      } catch (err) {
        console.error('Error checking model status:', err);
        // If error occurs but user is logged in, still set appropriate stats
        if (user) {
          setModelStats({
            is_trained: false,
            categories: categories
          });
        }
      } finally {
        setLoading(false);
      }
    };
    
    checkModel();
  }, [authChecked, user, auth]);

  // Add this at the beginning of the component
  useEffect(() => {
    // Log received transactions for debugging
    console.log('TransactionCategorizer received transactions:', userTransactions);
    
    if (userTransactions && userTransactions.length > 0) {
      // Check for uncategorized transactions
      const uncategorized = userTransactions.filter(
        tx => !tx.category || tx.category === '' || 
              tx.category === 'Uncategorized' || 
              tx.category === 'Other'
      );
      
      console.log(`Found ${uncategorized.length} uncategorized transactions out of ${userTransactions.length} total`);
      
      if (uncategorized.length > 0) {
        console.log('Uncategorized transactions:', uncategorized);
      }
    } else {
      console.warn('No transactions received or empty array');
    }
  }, [userTransactions]);

  // Handle categorize button click
  const handleCategorize = async () => {
    if (!sampleText.trim()) {
      setError('Please enter a transaction description');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      setPredictionResult(null);
      
      const postData = {
        description: sampleText
      };
      
      let config = {};
      
      // Only include user_id and auth token if logged in
      if (user) {
        postData.user_id = user.uid;
        
        // Get authentication token from Firebase Auth
        const currentUser = auth.currentUser;
        if (!currentUser) {
          throw new Error("No authenticated user found");
        }
        
        const idToken = await currentUser.getIdToken();
        config.headers = {
          'Authorization': `Bearer ${idToken}`
        };
      }
      
      const response = await axios.post('/api/categorizer/categorize', postData, config);
      
      if (response.data.status === 'success') {
        setPredictionResult(response.data.data);
        setSuccess(`Transaction categorized as "${response.data.data.category}"`);
      } else {
        setError(response.data.message || 'Failed to categorize transaction');
      }
    } catch (err) {
      console.error('Error categorizing transaction:', err);
      setError(err.response?.data?.message || 'Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  // Handle batch categorization of user transactions
// Updated handleBatchCategorize function for TransactionCategorizer component
const handleBatchCategorize = async () => {
  if (!user) {
    setError("You must be logged in to categorize transactions.");
    return;
  }
  
  if (!userTransactions || userTransactions.length === 0) {
    console.error('No transactions available to categorize');
    setError('No transactions available to categorize');
    return;
  }
  
  try {
    setLoading(true);
    setError(null);
    setPredictionResult(null);
    
    // Log the user transactions we're starting with for debugging
    console.log("Starting batch categorization with transactions:", userTransactions);
    
    // Filter transactions that don't have categories or are marked as Uncategorized or Other
    const uncategorizedTransactions = userTransactions.filter(
      tx => !tx.category || tx.category === '' || 
            tx.category === 'Uncategorized' || 
            tx.category === 'Other'
    );
    
    console.log(`Found ${uncategorizedTransactions.length} uncategorized transactions out of ${userTransactions.length} total`);
    
    if (uncategorizedTransactions.length === 0) {
      setSuccess('All transactions already have categories');
      setLoading(false);
      return;
    }
    
    // Get authentication token from Firebase Auth
    let idToken;
    try {
      const currentUser = auth.currentUser;
      if (!currentUser) {
        throw new Error("No authenticated user found");
      }
      
      idToken = await currentUser.getIdToken(true); // Force refresh token
      console.log("Successfully obtained Firebase ID token");
    } catch (tokenError) {
      console.error('Error getting authentication token:', tokenError);
      setError('Authentication error. Please try logging out and back in.');
      setLoading(false);
      return;
    }
    
    console.log(`Sending ${uncategorizedTransactions.length} transactions for categorization`);
    
    // Make the API request with proper error handling
    const response = await axios.post('/api/categorizer/batch', 
      {
        user_id: user.uid,
        transactions: uncategorizedTransactions
      },
      {
        headers: {
          'Authorization': `Bearer ${idToken}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    console.log("Batch categorization API response:", response.data);
    
    if (response.data.status === 'success') {
      const categorizedResults = response.data.data;
      
      if (!categorizedResults || categorizedResults.length === 0) {
        setError('No transactions were categorized by the API');
        setLoading(false);
        return;
      }
      
      setCategorizedTransactions(categorizedResults);
      setSuccess(`Categorized ${categorizedResults.length} transactions`);
      
      // Make a deep copy of the original transactions to avoid reference issues
      const updatedTransactions = JSON.parse(JSON.stringify(userTransactions));
      let updateCount = 0;
      
      // Log which fields are available in the categorized results
      console.log("Sample categorized result:", categorizedResults[0]);
      
      // Update each transaction based on the API response
      categorizedResults.forEach(catTx => {
        // Check if a transaction was categorized and has both an ID and category
        if (catTx.id) {
          // The category might be in different fields depending on the API response
          const newCategory = catTx.predicted_category || catTx.category;
          
          if (newCategory) {
            // Find the transaction in our copied array
            const index = updatedTransactions.findIndex(tx => tx.id === catTx.id);
            
            if (index !== -1) {
              // Debug which transaction is being updated
              console.log(`Updating transaction ${catTx.id} from "${updatedTransactions[index].category}" to "${newCategory}"`);
              
              // Update the category
              updatedTransactions[index].category = newCategory;
              updateCount++;
            } else {
              console.warn(`Transaction with ID ${catTx.id} not found in original list`);
            }
          } else {
            console.warn(`No category found for transaction ${catTx.id}`);
          }
        }
      });
      
      console.log(`Updated ${updateCount} transaction categories`);
      
      // Now pass the COMPLETE updated list back to the parent
      if (onCategorize) {
        console.log("Calling onCategorize with updated transactions");
        onCategorize(updatedTransactions);
      } else {
        console.warn("onCategorize callback is not defined");
      }
    } else {
      setError(response.data.message || 'Failed to categorize transactions');
    }
  } catch (err) {
    console.error('Error batch categorizing transactions:', err);
    if (err.response) {
      console.error('Response data:', err.response.data);
      console.error('Response status:', err.response.status);
      setError(`Server error (${err.response.status}): ${err.response.data.message || 'Unknown error'}`);
    } else if (err.request) {
      console.error('No response received:', err.request);
      setError('No response from server. Please check your network connection.');
    } else {
      setError(err.message || 'Error connecting to server');
    }
  } finally {
    setLoading(false);
  }
};

  // Handle training the model
  const handleTrain = async (trainingData) => {
    try {
      // Check if user is authenticated
      if (!user) {
        setError("You must be logged in to train the model. Please log in and try again.");
        setShowTrainingDialog(false);
        return;
      }
      
      setTraining(true);
      setError(null);
      
      // Get authentication token from Firebase Auth
      const currentUser = auth.currentUser;
      if (!currentUser) {
        throw new Error("No authenticated user found");
      }
      
      const idToken = await currentUser.getIdToken();
      
      const response = await axios.post('/api/categorizer/train', 
        {
          user_id: user.uid,
          transactions: trainingData
        },
        {
          headers: {
            'Authorization': `Bearer ${idToken}`
          }
        }
      );
      
      if (response.data.status === 'success') {
        setModelStats({
          is_trained: true,
          accuracy: response.data.data.accuracy,
          categories: response.data.data.categories,
          transaction_count: response.data.data.transaction_count
        });
        setSuccess('Model trained successfully! You can now categorize transactions automatically.');
      } else {
        setError(response.data.message || 'Failed to train model');
      }
    } catch (err) {
      console.error('Error training model:', err);
      setError(err.response?.data?.message || 'Error connecting to server');
    } finally {
      setTraining(false);
      setShowTrainingDialog(false);
    }
  };

  // Training Dialog component
  const TrainingDataDialog = () => {
    const [trainingData, setTrainingData] = useState([
      { description: 'AMAZON MKTPLACE', category: 'Shopping' },
      { description: 'UBER TRIP', category: 'Transportation' },
      { description: 'NETFLIX.COM', category: 'Entertainment' },
      { description: 'TRADER JOE\'S', category: 'Food & Dining' },
      { description: 'COMCAST CABLE', category: 'Utilities' },
      { description: 'CVS PHARMACY', category: 'Healthcare' },
      { description: 'SHELL OIL', category: 'Transportation' },
      { description: 'AMC THEATERS', category: 'Entertainment' },
      { description: 'AIRBNB', category: 'Travel' },
      { description: 'SALARY DEPOSIT', category: 'Income' },
      { description: 'WHOLE FOODS MARKET', category: 'Groceries' },
      { description: 'STATE FARM INSURANCE', category: 'Insurance' },
      { description: 'GREAT CLIPS', category: 'Personal Care' },
      { description: 'RED CROSS DONATION', category: 'Gifts & Donations' },
      { description: 'ROBINHOOD INVESTMENT', category: 'Investments' },
      { description: 'SPOTIFY PREMIUM', category: 'Subscriptions' },
      { description: 'IRS TAX PAYMENT', category: 'Taxes' },
      { description: 'OFFICE DEPOT', category: 'Business Expenses' },
      { description: 'BANK ATM FEE', category: 'Fees & Charges' },
      { description: 'COSTCO WHOLESALE', category: 'Groceries' },
      { description: 'GEICO AUTO INSURANCE', category: 'Insurance' },
      { description: 'SPA TREATMENT', category: 'Personal Care' },
      { description: 'UNICEF DONATION', category: 'Gifts & Donations' },
      { description: 'ETRADE INVESTMENT', category: 'Investments' },
      { description: 'DISNEY+ SUBSCRIPTION', category: 'Subscriptions' },
      { description: 'STATE TAX PAYMENT', category: 'Taxes' },
      { description: 'STAPLES OFFICE SUPPLIES', category: 'Business Expenses' },
      { description: 'CREDIT CARD FEE', category: 'Fees & Charges' }
    ]);
    const [newDescription, setNewDescription] = useState('');
    const [newCategory, setNewCategory] = useState('');
    
    const addExample = () => {
      if (!newDescription || !newCategory) return;
      
      setTrainingData([
        ...trainingData,
        { description: newDescription, category: newCategory }
      ]);
      
      setNewDescription('');
      setNewCategory('');
    };
    
    const removeExample = (index) => {
      const newData = [...trainingData];
      newData.splice(index, 1);
      setTrainingData(newData);
    };
    
    return (
      <Dialog 
        open={showTrainingDialog} 
        onClose={() => setShowTrainingDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <div className="flex items-center">
            <BrainCircuit className="mr-2 text-blue-600" />
            Train Transaction Categorizer
          </div>
        </DialogTitle>
        
        <DialogContent>
          <Typography variant="body2" className="mb-4">
            Provide example transactions with their correct categories to train the model.
            You need at least 20 examples for effective training.
          </Typography>
          
          {/* Add new example form */}
          <div className="flex gap-2 mb-4">
            <TextField
              label="Transaction Description"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              className="flex-grow"
              size="small"
            />
            <FormControl size="small" style={{ minWidth: 150 }}>
              <InputLabel>Category</InputLabel>
              <Select
                value={newCategory}
                label="Category"
                onChange={(e) => setNewCategory(e.target.value)}
              >
                {categories.map((cat) => (
                  <MenuItem key={cat} value={cat}>{cat}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button 
              variant="outlined" 
              onClick={addExample}
              disabled={!newDescription || !newCategory}
            >
              Add
            </Button>
          </div>
          
          {/* Examples table */}
          <TableContainer component={Paper} className="mb-4">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell style={{ width: '70%' }}>Description</TableCell>
                  <TableCell style={{ width: '20%' }}>Category</TableCell>
                  <TableCell style={{ width: '10%' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {trainingData.map((example, index) => (
                  <TableRow key={index}>
                    <TableCell>{example.description}</TableCell>
                    <TableCell>
                      <Chip 
                        label={example.category} 
                        size="small" 
                        color="primary" 
                        variant="outlined" 
                      />
                    </TableCell>
                    <TableCell>
                      <IconButton 
                        size="small" 
                        onClick={() => removeExample(index)}
                        color="error"
                      >
                        <X size={16} />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Alert severity="info" className="mb-4">
            <Typography variant="body2">
              <strong>Tips for good training data:</strong>
              <ul className="list-disc pl-5 mt-1">
                <li>Include a variety of transaction descriptions</li>
                <li>Make sure each category has multiple examples</li>
                <li>Use real transaction descriptions when possible</li>
                <li>Add variations of similar transactions</li>
              </ul>
            </Typography>
          </Alert>
          
          <Typography variant="body2" className="mt-2 text-right">
            {trainingData.length}/20 examples ({trainingData.length >= 20 ? 'sufficient' : 'need more'})
          </Typography>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowTrainingDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="primary"
            onClick={() => handleTrain(trainingData)}
            disabled={trainingData.length < 20 || training}
            startIcon={training ? <Loader className="animate-spin" /> : <BrainCircuit />}
          >
            {training ? 'Training...' : 'Train Model'}
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  // Determine if we should show authenticated UI or demo UI
  const isAuthenticated = user !== null && user !== undefined;

  return (
    <div className="space-y-6">
      {/* Authentication Debug Info */}
      {process.env.NODE_ENV === 'development' && (
        <Alert severity="info" className="mb-4">
          <Typography variant="subtitle2">Auth Status (Debug)</Typography>
          <Typography variant="body2">
            Auth Checked: {authChecked ? 'Yes' : 'No'}<br />
            Is Authenticated: {isAuthenticated ? 'Yes' : 'No'}<br />
            User ID: {user?.uid || 'None'}<br />
            Demo Mode: {isDemoMode ? 'Yes' : 'No'}
          </Typography>
        </Alert>
      )}

      {/* Demo Mode Notice */}
      {isDemoMode && !isAuthenticated && (
        <Alert severity="info" className="mb-4">
          <Typography variant="subtitle2">Demo Mode</Typography>
          <Typography variant="body2">
            You're using the categorizer in demo mode. Sign in to train a personalized model with your own data.
          </Typography>
        </Alert>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-4 flex items-center">
            <BrainCircuit className="mr-2 text-blue-600" />
            Smart Transaction Categorizer
          </Typography>
          
          {/* Model Status */}
          <div className="mb-6">
            <Typography variant="subtitle1" className="mb-2">Model Status:</Typography>
            {loading ? (
              <CircularProgress size={20} />
            ) : (
              <div className="flex items-center">
                {!isAuthenticated ? (
                  <Chip 
                    icon={<Info size={16} />}
                    label="Demo Mode" 
                    color="info" 
                    variant="outlined"
                  />
                ) : modelStats?.is_trained ? (
                  <Chip 
                    icon={<Check size={16} />}
                    label="Model Trained" 
                    color="success" 
                    variant="outlined"
                  />
                ) : (
                  <Chip 
                    icon={<AlertCircle size={16} />}
                    label="Model Not Trained" 
                    color="warning" 
                    variant="outlined"
                  />
                )}
                {modelStats?.accuracy && (
                  <Chip 
                    label={`Accuracy: ${(modelStats.accuracy * 100).toFixed(1)}%`}
                    color="primary"
                    variant="outlined"
                    className="ml-2"
                  />
                )}
              </div>
            )}
          </div>
          
          {/* Train Model Button - only show for logged in users */}
          {isAuthenticated && (
            <div className="mb-6">
              <Button
                variant="outlined"
                color="primary"
                startIcon={<BrainCircuit />}
                onClick={() => setShowTrainingDialog(true)}
                disabled={training}
              >
                {modelStats?.is_trained ? 'Retrain Model' : 'Train Model'}
              </Button>
              <Typography variant="body2" className="mt-1 text-gray-600">
                Train the AI to recognize your transaction patterns
              </Typography>
            </div>
          )}
          
          {/* Login prompt for demo mode */}
          {!isAuthenticated && (
            <div className="mb-6">
              <Button
                variant="contained"
                color="primary"
                onClick={() => window.location.href = '/login'}
              >
                Log in to Train Your Model
              </Button>
              <Typography variant="body2" className="mt-1 text-gray-600">
                Demo mode uses basic keyword matching. Sign in for ML-powered categorization.
              </Typography>
            </div>
          )}
          
          {/* Categorize Test Transaction */}
          <div className="mb-6">
            <Typography variant="subtitle1" className="mb-2">Test Categorization:</Typography>
            <div className="flex gap-2">
              <TextField
                label="Transaction Description"
                value={sampleText}
                onChange={(e) => setSampleText(e.target.value)}
                fullWidth
                margin="normal"
                className="flex-grow"
              />
              <Button
                variant="contained"
                color="primary"
                onClick={handleCategorize}
                disabled={loading}
                className="mt-4"
              >
                Categorize
              </Button>
            </div>
            
            {/* Prediction Result */}
            {predictionResult && (
              <Paper className="p-4 mt-3 bg-blue-50">
                <Typography variant="subtitle2">Category:</Typography>
                <div className="mt-1 flex items-center">
                  <Chip 
                    label={predictionResult.category} 
                    color="primary" 
                    className="mr-2"
                  />
                  <Typography variant="body2" className="text-gray-600">
                    {predictionResult.demo_mode 
                      ? "Demo mode (keyword matching)" 
                      : `Confidence: ${(predictionResult.confidence * 100).toFixed(0)}%`}
                  </Typography>
                </div>
              </Paper>
            )}
          </div>
          
          {/* Batch Categorization - only for logged in users */}
          {isAuthenticated && modelStats?.is_trained && (
            <div className="mb-3">
              <Typography variant="subtitle1" className="mb-2">Batch Categorize Transactions:</Typography>
              <Button
                variant="outlined"
                color="primary"
                startIcon={<Upload />}
                onClick={handleBatchCategorize}
                disabled={loading}
                className="mb-2"
              >
                Categorize Uncategorized Transactions
              </Button>
              <Typography variant="body2" className="text-gray-600">
                Automatically categorize all uncategorized transactions
              </Typography>
            </div>
          )}
          
          {/* Results display */}
          {success && (
            <Alert severity="success" className="mb-3" onClose={() => setSuccess(null)}>
              {success}
            </Alert>
          )}
          
          {error && (
            <Alert severity="error" className="mb-3" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
          
          {loading && (
            <LinearProgress className="mb-3" />
          )}
          
          {/* Recently Categorized Transactions */}
          {categorizedTransactions.length > 0 && (
            <Box className="mt-4">
              <Typography variant="subtitle1" className="mb-2">
                Recently Categorized Transactions:
              </Typography>
              <TableContainer component={Paper} style={{ maxHeight: 300 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Description</TableCell>
                      <TableCell>Category</TableCell>
                      <TableCell>Confidence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {categorizedTransactions.slice(0, 10).map((tx, index) => (
                      <TableRow key={index}>
                        <TableCell>{tx.description}</TableCell>
                        <TableCell>
                          <Chip 
                            label={tx.predicted_category} 
                            size="small" 
                            color="primary" 
                          />
                        </TableCell>
                        <TableCell>
                          {tx.category_confidence ? 
                            `${(tx.category_confidence * 100).toFixed(0)}%` : 
                            'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </CardContent>
      </Card>
      
      {/* How It Works */}
      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-3 flex items-center">
            <Info className="mr-2 text-blue-600" />
            How It Works
          </Typography>
          
          <Typography variant="body2" className="mb-3">
            The Smart Transaction Categorizer uses Natural Language Processing (NLP) to automatically
            categorize your transactions based on their descriptions.
          </Typography>
          
          <Typography variant="subtitle2" className="mt-4 mb-1">The process:</Typography>
          <ol className="list-decimal ml-6 mb-4">
            <li className="mb-1">Train the model with example transactions</li>
            <li className="mb-1">The AI learns patterns in transaction descriptions</li>
            <li className="mb-1">The model categorizes new transactions automatically</li>
            <li className="mb-1">Results include confidence scores for transparency</li>
          </ol>
          
          <Typography variant="body2" className="mt-3">
            This saves you time on manual categorization and helps ensure consistent
            categorization across all your transactions.
          </Typography>
        </CardContent>
      </Card>
      
      {/* Training Dialog - only available for logged in users */}
      {isAuthenticated && <TrainingDataDialog />}
    </div>
  );
};

export default TransactionCategorizer; 