import React, { useState } from "react";
import { Button, Form } from "react-bootstrap";
import { useUserAuth } from "../context/UserAuthContext";
import { db } from "../firebase"; // Firebase setup for Firestore
import { collection, addDoc } from "firebase/firestore"; // Firestore methods
import { useNavigate } from "react-router-dom"; // Import useNavigate hook

const Home = () => {
  const { user } = useUserAuth();
  const [name, setName] = useState("");
  const [mobile, setMobile] = useState("");
  const [speciesList, setSpeciesList] = useState([{ species: "", count: "" }]);
  const navigate = useNavigate(); // Initialize navigate

  // Add more species input
  const handleAddSpecies = () => {
    setSpeciesList([...speciesList, { species: "", count: "" }]);
  };

  // Handle species input change
  const handleSpeciesChange = (index, e) => {
    const { name, value } = e.target;
    const newSpeciesList = [...speciesList];
    newSpeciesList[index][name] = value;
    setSpeciesList(newSpeciesList);
  };

  // Handle form submission
  // Handle form submission
const handleSubmit = async (e) => {
  e.preventDefault();
  try {
    // Store form data in Firebase under the user's ID
    await addDoc(collection(db, "users", user.uid, "animalData"), {
      name,
      mobile,
      speciesList,
    });

    // Prepare expected counts from speciesList
    const expectedCounts = {};
    speciesList.forEach(speciesData => {
      expectedCounts[speciesData.species] = parseInt(speciesData.count) || 0;
    });

    // Generate a message showing animal names and their counts
    const speciesInfo = speciesList.map(speciesData => `${speciesData.species}: ${speciesData.count}`).join(", ");
    alert(`Form data submitted successfully! Animal Info: ${speciesInfo}`);

    // Navigate to the detection page after successful form submission, passing mobile and expectedCounts
    navigate("/detect", { state: { mobileNumber: mobile, expectedCounts } });
  } catch (error) {
    console.error("Error adding document: ", error);
  }
};


  return (
    <div className="p-4 box">
      <h2 className="mb-3">Animal Information Form</h2>
      <Form onSubmit={handleSubmit}>
        <Form.Group className="mb-3" controlId="formName">
          <Form.Label>Name</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </Form.Group>

        <Form.Group className="mb-3" controlId="formMobile">
          <Form.Label>Mobile Number</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter your mobile number"
            value={mobile}
            onChange={(e) => setMobile(e.target.value)}
            required
          />
        </Form.Group>

        {speciesList.map((speciesData, index) => (
          <div key={index}>
            <Form.Group className="mb-3" controlId={`formSpecies${index}`}>
              <Form.Label>Animal Species</Form.Label>
              <Form.Control
                type="text"
                placeholder="Enter animal species"
                name="species"
                value={speciesData.species}
                onChange={(e) => handleSpeciesChange(index, e)}
                required
              />
            </Form.Group>

            <Form.Group className="mb-3" controlId={`formCount${index}`}>
              <Form.Label>Animal Count</Form.Label>
              <Form.Control
                type="number"
                placeholder="Enter animal count"
                name="count"
                value={speciesData.count}
                onChange={(e) => handleSpeciesChange(index, e)}
                required
              />
            </Form.Group>
          </div>
        ))}

        <div className="d-grid gap-2">
          <Button variant="secondary" onClick={handleAddSpecies}>
            Add More Species
          </Button>

          <Button variant="primary" type="submit">
            Submit
          </Button>
        </div>
      </Form>
    </div>
  );
};

export default Home;
