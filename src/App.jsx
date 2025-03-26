  import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
  import { ThemeProvider, createTheme } from '@mui/material/styles';
  import { AuthProvider, useAuth } from './contexts/AuthContext';
  import Login from './components/Login';
  import Signup from './components/Signup';
  import Dashboard from './components/Dashboard';
  import Profile from './components/Profile';
  import AIInsights from './components/AIInsights';
  import FinanceNews from './components/FinanceNews';

  function App() {
    const theme = createTheme({
      palette: {
        primary: {
          main: '#1976d2',
        },
        secondary: {
          main: '#dc2626',
        },
      },
    });

    const PrivateRoute = ({ children }) => {
      const { user } = useAuth();
      return user ? children : <Navigate to="/login" />;
    };

    return (
      <ThemeProvider theme={theme}>
        <AuthProvider>
          <Router>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
              <Route
                path="/dashboard"
                element={
                  <PrivateRoute>
                    <Dashboard />
                  </PrivateRoute>
                }
              />
              <Route
                path="/profile"
                element={
                  <PrivateRoute>
                    <Profile />
                  </PrivateRoute>
                }
              />
              <Route
                path="/ai-insights"
                element={
                  <PrivateRoute>
                    <AIInsights />
                  </PrivateRoute>
                }
              />
              <Route path="/" element={<Navigate to="/login" />} />
              <Route path = "/finance-news" element = {<FinanceNews />} />
            </Routes>
          </Router>
        </AuthProvider>
      </ThemeProvider>
    );
  }

  export default App;