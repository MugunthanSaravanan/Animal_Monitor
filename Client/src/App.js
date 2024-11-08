import { Container } from "react-bootstrap";
import { Routes, Route } from "react-router-dom";
import "./App.css";
import Home from "./components/Home";
import Login from "./components/Login";
import Signup from "./components/Signup";
import AnimalDetection from "./components/Detection"; // Updated to match the new detection component name
import ProtectedRoute from "./components/ProtectedRoute";
import { UserAuthContextProvider } from "./context/UserAuthContext";
import AboutUs from "./components/AboutUs"; // Importing the AboutUs page

function App() {
  return (
    <Container>
      <UserAuthContextProvider>
        <Routes>
          {/* Protected routes */}
          <Route
            path="/home"
            element={
              <ProtectedRoute>
                <Home />
              </ProtectedRoute>
            }
          />
          <Route
            path="/detect"
            element={
              <ProtectedRoute>
                <AnimalDetection />
              </ProtectedRoute>
            }
          />

          {/* Public routes */}
          <Route path="/" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* About Us page route */}
          <Route path="/about" element={<AboutUs />} />
        </Routes>
      </UserAuthContextProvider>
    </Container>
  );
}

export default App;
