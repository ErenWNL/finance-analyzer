import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent,
  TextField, 
  Button, 
  Typography, 
  Alert,
  Link,
  InputAdornment,
  IconButton,
  Box,
  CircularProgress
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Visibility, VisibilityOff, Email, Lock } from '@mui/icons-material';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();

  // Show success message if redirected from signup
  const [successMessage, setSuccessMessage] = useState('');
  
  useEffect(() => {
    if (location.state?.message) {
      setSuccessMessage(location.state.message);
    }
  }, [location]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setError('');
      setLoading(true);
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError('Invalid email or password. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePassword = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <Card className="max-w-md w-full shadow-lg rounded-lg">
        <CardContent className="p-6">
          <div className="text-center mb-6">
            <Typography component="h1" variant="h5" className="font-medium">
              Sign in to your account
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
          
          {successMessage && (
            <Alert 
              severity="success" 
              className="mb-4"
              onClose={() => setSuccessMessage('')}
            >
              {successMessage}
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
            
            <TextField
              label="Password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              fullWidth
              autoComplete="current-password"
              variant="outlined"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={handleTogglePassword}
                      edge="end"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            
            <div className="text-right">
              <Link href="/forgot-password" variant="body2" className="text-blue-600 hover:text-blue-800">
                Forgot password?
              </Link>
            </div>
            
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
                "Sign In"
              )}
            </Button>
            
            <div className="text-center mt-4">
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

export default Login;