import React, { createContext, useState, useContext, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile,
  sendPasswordResetEmail
} from 'firebase/auth';
import { auth, db } from '../config/firebase';
import { doc, setDoc, getDoc } from 'firebase/firestore';

const AuthContext = createContext({});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Enhanced signup with profile creation and expenses initialization
  const signup = async (email, password, displayName) => {
    try {
      const { user } = await createUserWithEmailAndPassword(auth, email, password);
      
      // Update profile with display name if provided
      if (displayName) {
        await updateProfile(user, { displayName });
      }

      // Create user document in Firestore with initial expenses array
      await setDoc(doc(db, 'users', user.uid), {
        email: user.email,
        displayName: displayName || '',
        createdAt: new Date().toISOString(),
        lastLogin: new Date().toISOString(),
        expenses: []  // Initialize empty expenses array
      });

      return user;
    } catch (error) {
      throw error;
    }
  };

  // Enhanced login with last login update and expenses check
  const login = async (email, password) => {
    try {
      const result = await signInWithEmailAndPassword(auth, email, password);
      
      const userRef = doc(db, 'users', result.user.uid);
      const userDoc = await getDoc(userRef);

      // If user document exists, update last login
      if (userDoc.exists()) {
        await setDoc(userRef, {
          lastLogin: new Date().toISOString()
        }, { merge: true });
      } else {
        // If user document doesn't exist (rare case), create it
        await setDoc(userRef, {
          email: result.user.email,
          displayName: result.user.displayName || '',
          createdAt: new Date().toISOString(),
          lastLogin: new Date().toISOString(),
          expenses: []
        });
      }

      return result;
    } catch (error) {
      throw error;
    }
  };

  // Enhanced logout with proper cleanup
  const logout = async () => {
    try {
      if (user) {
        // No need to clear localStorage as we're using Firestore now
        await signOut(auth);
      }
    } catch (error) {
      throw error;
    }
  };

  // Password reset functionality
  const resetPassword = (email) => {
    return sendPasswordResetEmail(auth, email);
  };

  // Update user profile with Firestore sync
  const updateUserProfile = async (data) => {
    try {
      if (!user) throw new Error('No user logged in');

      // Update Firebase Auth profile
      await updateProfile(user, data);
      
      // Update Firestore user document
      const userRef = doc(db, 'users', user.uid);
      await setDoc(userRef, {
        ...data,
        updatedAt: new Date().toISOString()
      }, { merge: true });

      // Get updated user data
      const updatedDoc = await getDoc(userRef);
      return updatedDoc.data();
    } catch (error) {
      throw error;
    }
  };

  // Get user data from Firestore
  const getUserData = async () => {
    try {
      if (!user) throw new Error('No user logged in');

      const userRef = doc(db, 'users', user.uid);
      const docSnap = await getDoc(userRef);

      if (docSnap.exists()) {
        return docSnap.data();
      } else {
        throw new Error('User data not found');
      }
    } catch (error) {
      throw error;
    }
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        // Get user data from Firestore when auth state changes
        try {
          const userRef = doc(db, 'users', currentUser.uid);
          const docSnap = await getDoc(userRef);
          if (docSnap.exists()) {
            setUser({ ...currentUser, ...docSnap.data() });
          } else {
            setUser(currentUser);
          }
        } catch (error) {
          console.error('Error fetching user data:', error);
          setUser(currentUser);
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const value = {
    user,
    signup,
    login,
    logout,
    resetPassword,
    updateUserProfile,
    getUserData,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};