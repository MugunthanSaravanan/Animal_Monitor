// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore"; // Firestore SDK

// Your Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAXxIJ_oR0GFCBFcgzpGH8qdFQPLDBqe2s",
  authDomain: "animal-ce4c9.firebaseapp.com",
  projectId: "animal-ce4c9",
  storageBucket: "animal-ce4c9.appspot.com",
  messagingSenderId: "313895207593",
  appId: "1:313895207593:web:ee0733b42968afd7641fb8",
  measurementId: "G-S461R398LC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth
export const auth = getAuth(app);

// Initialize Firestore
export const db = getFirestore(app); // Export Firestore

export default app;
