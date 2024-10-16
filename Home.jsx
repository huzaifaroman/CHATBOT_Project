'use client'; // Make page.jsx a Client Component

import { useState } from 'react';

export default function Home() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');

  const sendMessage = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }), // Ensure the message is sent as JSON
      });
      if (!res.ok) {
        throw new Error(`Server Error: ${res.statusText}`);
      }
      const data = await res.json();
      if (data.text) {
        setResponse(data.text); // Update response from the server
      } else {
        throw new Error('Invalid response from server.');
      }
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    }
  };

  return (
    <div>
      <h1>Welcome to My Chatbot App</h1>
      <input 
        type="text" 
        value={message} 
        onChange={(e) => setMessage(e.target.value)} 
        placeholder="Enter your message" 
      />
      <button onClick={sendMessage}>Send Message</button>

      {response && <div><strong>Bot Response:</strong> {response}</div>}
    </div>
  );
}
