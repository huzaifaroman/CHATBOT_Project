import React, { useState } from 'react'; // Add this line

const ChatInput = ({ onSendMessage, isPdfMode }) => {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    onSendMessage(inputText, isPdfMode);
    setInputText('');
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-white border-t">
      <input
        type="text"
        placeholder={
          isPdfMode ? 'Ask about the PDF...' : 'Ask any question...'
        }
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        className="w-full p-2 border rounded-md focus:outline-none focus:ring focus:ring-blue-300"
      />
      <button
        type="submit"
        className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none"
      >
        Send
      </button>
    </form>
  );
};

export default ChatInput;
