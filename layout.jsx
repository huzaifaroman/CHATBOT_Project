'use client'; // Mark as Client Component

import { useState } from 'react';
import localFont from "next/font/local";
import Sidebar from '../components/Sidebar';
import Chat from '../components/Chat';

import "./globals.css"; // Make sure to import your global CSS

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});

const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

// Remove metadata export from here

export default function RootLayout({ children }) { 
  const [isChatVisible, setIsChatVisible] = useState(false); 

  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <div className="flex">
          <Sidebar setIsChatVisible={setIsChatVisible} />
          <main className="flex-grow">
            {children} 
          </main>
          {isChatVisible && <Chat />} {/* Make sure Chat is imported if used here */}
        </div>
      </body>
    </html>
  );
}
