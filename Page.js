import Sidebar from '@/components/Sidebar'; 
import Chat from '@/components/Chat';

export default function Home() {
  return (
    <main className="flex h-screen">
      <Sidebar /> {/* Optional sidebar */}
      <Chat />
    </main>
  );
}