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
import { auth } from '../config/firebase';

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

  // Create axios instance with default config
  const api = axios.create({
    baseURL: 'http://localhost:5001',
    withCredentials: true,
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }
  });

  useEffect(() => {
    loadProfile();
  }, [user]);

  const loadProfile = async () => {
    try {
      const user = auth.currentUser;
      if (!user) {
        console.error('No authenticated user found');
        return;
      }

      // Get user profile from Firestore
      const profileDoc = await getDoc(doc(db, 'users', user.uid));
      if (profileDoc.exists()) {
        const profileData = profileDoc.data();
        setProfileData(prevData => ({
          ...prevData,
          ...profileData
        }));

        // Load profile photo from backend
        try {
          const token = await user.getIdToken();
          const response = await api.get(`/api/profile/photo/${user.uid}`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Accept': 'application/json'
            }
          });
          
          if (response.data.photo_data) {
            // Create a data URL from the base64 photo data
            const photoUrl = `data:${response.data.content_type};base64,${response.data.photo_data}`;
            setProfileData(prev => ({ ...prev, photoURL: photoUrl }));
          }
        } catch (error) {
          console.error('Error loading profile photo:', error);
          // Don't throw error here, just log it
        }
      }
    } catch (error) {
      console.error('Error loading profile:', error);
      setError('Failed to load profile. Please try again.');
    }
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const user = auth.currentUser;
      if (!user) {
        throw new Error('No authenticated user found');
      }

      const formData = new FormData();
      formData.append('photo', file);

      const token = await user.getIdToken();
      const response = await api.post(
        `/api/profile/photo`,
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      if (response.data.success) {
        // Reload the profile to get the updated photo
        await loadProfile();
        setSuccess('Photo uploaded successfully!');
      }
    } catch (error) {
      console.error('Error uploading photo:', error);
      setError('Failed to upload photo. Please try again.');
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
                    src={profileData.photoURL || 'https://ui-avatars.com/api/?name=' + encodeURIComponent(profileData.displayName || 'User') + '&background=random'}
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