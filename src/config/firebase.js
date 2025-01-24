import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';


const firebaseConfig = {
    apiKey: "AIzaSyCbs8HfmF8UO9v3WvFBZCH7aJmTqHBPgiU",
    authDomain: "finance-analyzer-15afe.firebaseapp.com",
    projectId: "finance-analyzer-15afe",
    storageBucket: "finance-analyzer-15afe.firebasestorage.app",
    messagingSenderId: "630711742274",
    appId: "1:630711742274:web:46834a723e72c9ac5e22db",
    measurementId: "G-VKE5YJP87S"
  };

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const firebaseApp = app;
export const db = getFirestore(app);
export const storage = getStorage(app);
