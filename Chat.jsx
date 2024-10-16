'use client';
import { useState } from 'react';
import ChatInput from './ChatInput';
import Message from './Message';
import PdfUploader from './PdfUploader';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [isPdfMode, setIsPdfMode] = useState(false);
  const [pdfPath, setPdfPath] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (messageText, isPdfQuery) => {
    if (isPdfQuery && !pdfPath) {
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: 'Please upload a PDF before sending a query.', sender: 'bot' },
      ]);
      return;
    }

    setIsLoading(true);
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: messageText, sender: 'user' },
    ]);

    try {
      const response = await fetch(
        isPdfQuery ? 'http://127.0.0.1:5000/api/pdf-query' : 'http://127.0.0.1:5000/api/chat',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: messageText,
            pdf_path: isPdfQuery ? pdfPath : undefined,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Server Error: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.response) {
        const botMessage = {
          text: data.response,
          timestamp: new Date().toLocaleString(),
          sender: 'bot',
        };

        setMessages((prevMessages) => [
          ...prevMessages,
          botMessage,
        ]);
      } else {
        throw new Error('Invalid response from server.');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      const errorResponseMessage = {
        text: `Error: ${errorMessage}`,
        timestamp: new Date().toLocaleString(),
        sender: 'bot',
      };

      setMessages((prevMessages) => [
        ...prevMessages,
        errorResponseMessage,
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePdfUpload = (path) => {
    setPdfPath(path);
    setIsPdfMode(true);
  };

  return (
    <div className="fixed bottom-0 right-0 w-full sm:w-[400px] h-[500px] bg-white border-t shadow-xl flex flex-col"> 
      <div className="flex items-center justify-between p-3 bg-gray-200">
        <h3 className="font-semibold">Chatbot</h3>
      </div>
      <div className="flex-grow p-4 overflow-y-auto">
        {messages.map((message, index) => (
          <Message key={index} text={message.text} sender={message.sender} />
        ))}
        {isLoading && <p className="text-center">Loading...</p>}
      </div>
      {!isPdfMode && <PdfUploader onPdfUpload={handlePdfUpload} />}
      <ChatInput onSendMessage={handleSendMessage} isPdfMode={isPdfMode} />
    </div>
  );
};

export default Chat;
