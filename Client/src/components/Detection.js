import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import './Detection.css';

function Detection() {
  const location = useLocation();
  const { mobileNumber, expectedCounts } = location.state || { mobileNumber: '', expectedCounts: {} };

  const [animalCounts, setAnimalCounts] = useState({});
  const [uploadMessage, setUploadMessage] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing the camera:', error);
        setUploadMessage('Error accessing the camera. Please check your device settings.');
      }
    };

    startCamera();

    const captureFrame = async () => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        const blob = await fetch(dataURL).then(res => res.blob());

        const formData = new FormData();
        formData.append('file', blob, 'image.jpg');

        const additionalData = {
          mobile_number: mobileNumber,
          expected_counts: expectedCounts,
        };
        formData.append('form_data', JSON.stringify(additionalData));

        try {
          const res = await axios.post('http://127.0.0.1:5000/detect', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          console.log('Response data:', res.data);
          setAnimalCounts(res.data || {});
          setUploadMessage('');  // Clear any previous messages
        } catch (error) {
          console.error('Error fetching animal counts:', error);
          setUploadMessage(`Error: ${error.response?.data?.error || error.message}`);
        }
      }
    };

    intervalRef.current = setInterval(captureFrame, 2000);

    return () => clearInterval(intervalRef.current);
  }, [mobileNumber, expectedCounts]);

  return (
    <div className="container">
      <div className="camera-preview">
        <video ref={videoRef} autoPlay style={{ width: '100%', height: 'auto' }} />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
      <div className="animal-counts">
        <h2>Animal Counts:</h2>
        <table>
          <thead>
            <tr>
              <th>Animal</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(animalCounts).length === 0 ? (
              <tr><td colSpan="2">No data available</td></tr>
            ) : (
              Object.entries(animalCounts).map(([animal, count]) => (
                <tr key={animal}>
                  <td>{animal}</td>
                  <td>{count}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
        {uploadMessage && <p>{uploadMessage}</p>}
      </div>
    </div>
  );
}

export default Detection;
