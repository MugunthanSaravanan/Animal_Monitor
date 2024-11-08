import React from "react";
import './AboutUs.css'; // Assuming you'll style it in a separate CSS file

import mugunthanImage from "../images/mugunthan.jpg"; 
import madhankumarImage from "../images/madhankumar.png";
import riyaImage from "../images/riyas.png";
import karthikeyanImage from "../images/karthikeyan.png";

const teamMembers = [
  {
    name: "Mugunthan S",
    role: "Front end",
    link: "https://mugunthan.xyz",
    image: mugunthanImage
  },
  {
    name: "Madhankumar S",
    role: "Model",
    link: "#",
    image: madhankumarImage
  },
  {
    name: "Mohamed Riyas S",
    role: "Back end",
    link: "#",
    image: riyaImage
  },
  {
    name: "Karthikeyan S",
    role: "Dataset and Realtime image collection",
    link: "#",
    image: karthikeyanImage
  }
];

const AboutUs = () => {
  return (
    <div className="about-container">
      <h1>Meet Our Team</h1>
      <div className="team-grid">
        {teamMembers.map((member, index) => (
          <div className="team-member" key={index}>
            <img src={member.image} alt={member.name} className="profile-pic" />
            <h3>{member.name}</h3>
            <p>{member.role}</p>
            <a href={member.link} target="_blank" rel="noopener noreferrer">
              {member.link !== "#" ? "Website" : ""}
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AboutUs;
