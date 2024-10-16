const Message = ({ text, sender }) => {
    const isUser = sender === 'user';
  
    return (
      <div
        className={`mb-2 p-3 rounded-lg max-w-xs ${
          isUser
            ? 'bg-blue-200 text-right ml-auto'
            : 'bg-gray-300 text-left mr-auto'
        }`}
      >
        {text}
      </div>
    );
  };
  
  export default Message;