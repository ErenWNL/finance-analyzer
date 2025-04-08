import React, { useState } from 'react';
import { 
  Card, 
  CardContent,
  TextField, 
  Button, 
  Typography, 
  Alert,
  Link,
  InputAdornment,
  Box,
  CircularProgress
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Email } from '@mui/icons-material';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { resetPassword } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setMessage('');
      setError('');
      setLoading(true);
      await resetPassword(email);
      setMessage('Check your email inbox for password reset instructions');
    } catch (err) {
      setError('Failed to reset password. Please check if the email is correct.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <Card className="max-w-md w-full shadow-lg rounded-lg">
        <CardContent className="p-6">
          <div className="text-center mb-6">
            <Typography component="h1" variant="h5" className="font-medium">
              Reset Password
            </Typography>
            <Typography variant="body2" color="textSecondary" className="mt-2">
              Enter your email and we'll send you instructions to reset your password
            </Typography>
          </div>
          
          {error && (
            <Alert 
              severity="error" 
              className="mb-4"
              onClose={() => setError('')}
            >
              {error}
            </Alert>
          )}
          
          {message && (
            <Alert 
              severity="success" 
              className="mb-4"
              onClose={() => setMessage('')}
            >
              {message}
            </Alert>
          )}
          
          <form className="space-y-4" onSubmit={handleSubmit}>
            <TextField
              label="Email Address"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              fullWidth
              autoComplete="email"
              variant="outlined"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email color="action" />
                  </InputAdornment>
                ),
              }}
            />
            
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              disabled={loading}
              className="mt-4 py-2"
              disableElevation
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                "Reset Password"
              )}
            </Button>
            
            <div className="flex justify-between items-center mt-4">
              <Link 
                href="/login" 
                variant="body2"
                className="text-blue-600 hover:text-blue-800"
              >
                Back to Login
              </Link>
              
              <Typography variant="body2" color="textSecondary">
                Don't have an account?{' '}
                <Link 
                  href="/signup" 
                  variant="body2"
                  className="text-blue-600 hover:text-blue-800"
                >
                  Sign Up
                </Link>
              </Typography>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default ForgotPassword;