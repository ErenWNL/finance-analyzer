import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  TextField, 
  Button, 
  Avatar, 
  Typography,
  Alert,
  CircularProgress,
  Box,
  IconButton
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { getFirestore, doc, setDoc, getDoc } from 'firebase/firestore';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Camera } from 'lucide-react';
import axios from 'axios';

const Profile = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Add state declarations
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [profileData, setProfileData] = useState({
    displayName: '',
    age: '',
    bio: '',
    phoneNumber: '',
    photoURL: ''
  });

  const db = getFirestore();

  useEffect(() => {
    loadProfile();
  }, [user]);

  const loadProfile = async () => {
    if (!user) return;

    try {
      setLoading(true);
      const docRef = doc(db, 'users', user.uid);
      const docSnap = await getDoc(docRef);
      
      if (docSnap.exists()) {
        const data = docSnap.data();
        // Try to load profile photo from MongoDB
        try {
          const response = await axios.get(`http://localhost:5000/api/profile/photo/${user.uid}`, {
            withCredentials: true,
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }
          });
          if (response.data.photo) {
            data.photoURL = `data:image/jpeg;base64,${response.data.photo}`;
          }
        } catch (err) {
          console.error('Error loading profile photo:', err);
        }
        
        setProfileData(prevData => ({
          ...prevData,
          ...data
        }));
      }
    } catch (err) {
      console.error('Error loading profile:', err);
      setError('Failed to load profile data');
    } finally {
      setLoading(false);
    }
  };

  const handlePhotoUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !user) return;

    try {
      setLoading(true);
      setError('');
      
      const formData = new FormData();
      formData.append('photo', file);
      formData.append('userId', user.uid);

      const response = await axios.post('http://localhost:5000/api/profile/photo', formData, {
        withCredentials: true,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.photo) {
        const photoURL = `data:image/jpeg;base64,${response.data.photo}`;
        setProfileData(prev => ({ ...prev, photoURL }));
        setSuccess('Photo uploaded successfully!');
      }
    } catch (err) {
      console.error('Error uploading photo:', err);
      setError('Failed to upload photo. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!user) return;

    try {
      setLoading(true);
      setError('');
      setSuccess('');

      const userRef = doc(db, 'users', user.uid);
      await setDoc(userRef, profileData, { merge: true });
      setSuccess('Profile updated successfully!');
    } catch (err) {
      console.error('Error updating profile:', err);
      setError('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box className="min-h-screen bg-gray-100 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        <Button
          variant="text"
          color="primary"
          startIcon={<ArrowLeft />}
          onClick={() => navigate('/dashboard')}
          className="mb-8"
          style={{ textTransform: 'none' }}
        >
          Back to Dashboard
        </Button>

        <Card elevation={3}>
          <CardContent className="space-y-8 p-8">
            <Typography 
              variant="h5" 
              component="h2" 
              align="center" 
              className="mb-8"
              style={{ fontWeight: 500 }}
            >
              Profile Settings
            </Typography>

            {error && <Alert severity="error" className="mb-6">{error}</Alert>}
            {success && <Alert severity="success" className="mb-6">{success}</Alert>}

            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Avatar Upload with improved styling */}
              <Box className="flex justify-center mb-12">
                <div className="relative">
                  <Avatar
                    src={profileData.photoURL}
                    alt={profileData.displayName || 'Profile'}
                    sx={{ 
                      width: 120, 
                      height: 120,
                      border: '4px solid #fff',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                  />
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handlePhotoUpload}
                    className="hidden"
                    id="photo-upload"
                  />
                  <label htmlFor="photo-upload">
                    <IconButton
                      component="span"
                      sx={{
                        position: 'absolute',
                        bottom: 0,
                        right: 0,
                        backgroundColor: '#fff',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        '&:hover': {
                          backgroundColor: '#f5f5f5',
                        },
                      }}
                    >
                      <Camera size={20} />
                    </IconButton>
                  </label>
                </div>
              </Box>

              <div className="space-y-6">
                <TextField
                  fullWidth
                  label="Display Name"
                  value={profileData.displayName}
                  onChange={(e) => setProfileData(prev => ({ ...prev, displayName: e.target.value }))}
                  variant="outlined"
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Age"
                  type="number"
                  value={profileData.age}
                  onChange={(e) => setProfileData(prev => ({ ...prev, age: e.target.value }))}
                  variant="outlined"
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Bio"
                  multiline
                  rows={4}
                  value={profileData.bio}
                  onChange={(e) => setProfileData(prev => ({ ...prev, bio: e.target.value }))}
                  variant="outlined"
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Phone Number"
                  value={profileData.phoneNumber}
                  onChange={(e) => setProfileData(prev => ({ ...prev, phoneNumber: e.target.value }))}
                  variant="outlined"
                  sx={{ mb: 4 }}
                />
              </div>

              <Box className="flex justify-center pt-6">
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={loading}
                  sx={{ 
                    minWidth: '200px',
                    height: '48px',
                    borderRadius: '24px',
                    textTransform: 'none',
                    fontSize: '1rem'
                  }}
                >
                  {loading ? <CircularProgress size={24} /> : 'Save Profile'}
                </Button>
              </Box>
            </form>
          </CardContent>
        </Card>
      </div>
    </Box>
  );
};

export default Profile;