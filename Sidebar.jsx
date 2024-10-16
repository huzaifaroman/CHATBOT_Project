'use client'
import Link from 'next/link';

const Sidebar = ({ setIsChatVisible }) => {
  return (
    <aside className="bg-blue-500 w-64 h-screen p-4 text-white fixed top-0 left-0">
      <h1 className="text-2xl font-bold mb-4">PDF Chatbot</h1>
      <ul>
        <li className="mb-2">
          <Link href="/" onClick={() => setIsChatVisible(false)} className="hover:text-gray-200">
            Home
          </Link>
        </li>
        <li className="mb-2"> 
          <button onClick={() => setIsChatVisible(true)} className="hover:text-gray-200">
            Chat
          </button>
        </li>
        {/* ... More links ... */}
      </ul>
    </aside>
  );
};

export default Sidebar;