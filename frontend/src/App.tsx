import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Bot, User, Terminal, Cpu, Settings, MessageSquare, Volume2, VolumeX } from 'lucide-react';
import Markdown from 'markdown-to-jsx';
import Dashboard from './Dashboard';
import './App.css';

interface Message {
  role: 'user' | 'ai';
  content: string;
  mood?: string;
  intimacy?: number;
}

// =============================================================================
// API Configuration - Uses relative URLs, nginx handles routing
// =============================================================================
const API_BASE = import.meta.env.VITE_API_URL || '/api';
const API_KEY = import.meta.env.VITE_API_KEY || '';

// Helper for authenticated fetch
const authFetch = (url: string, options: RequestInit = {}) => {
  const headers = new Headers(options.headers);
  if (API_KEY) {
    headers.set('X-API-Key', API_KEY);
  }
  headers.set('Content-Type', 'application/json');
  return fetch(url, { ...options, headers });
};

function App() {
  const [view, setView] = useState<'chat' | 'dashboard'>('chat');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [project] = useState('global'); 
  const [status, setStatus] = useState('Connecting...');
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  // Speak AI message using ElevenLabs
  const speakMessage = useCallback(async (text: string, index: number) => {
    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    
    // If clicking the same message that's playing, just stop
    if (speakingIndex === index) {
      setSpeakingIndex(null);
      return;
    }
    
    setSpeakingIndex(index);
    
    try {
      const response = await authFetch(`${API_BASE}/voice/speak`, {
        method: 'POST',
        body: JSON.stringify({ message: text }),
      });
      
      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioRef.current = audio;
        
        audio.onended = () => {
          setSpeakingIndex(null);
          URL.revokeObjectURL(audioUrl);
          audioRef.current = null;
        };
        
        audio.onerror = () => {
          setSpeakingIndex(null);
          URL.revokeObjectURL(audioUrl);
          audioRef.current = null;
        };
        
        audio.play();
      } else {
        setSpeakingIndex(null);
        console.error('Voice API error:', response.status);
      }
    } catch (error) {
      setSpeakingIndex(null);
      console.error('Voice playback failed:', error);
    }
  }, [speakingIndex]);
  
  // 1. Heartbeat Polling (uses relative URL)
  useEffect(() => {
    const pollHeartbeat = async () => {
      try {
        const res = await authFetch(`${API_BASE}/notifications`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.notifications && data.notifications.length > 0) {
          data.notifications.forEach((note: unknown) => {
            // Handle both string and object notifications
            const message = typeof note === 'string' 
              ? note 
              : (note as {message?: string}).message || 'New notification';
            
            setMessages(prev => [...prev, { 
              role: 'ai', 
              content: `🔔 **Notification:** ${message}`, 
              mood: 'proactive' 
            }]);
          });
        }
      } catch {
        // Silent fail - backend might be restarting
      }
    };

    const interval = setInterval(pollHeartbeat, 10000);
    return () => clearInterval(interval);
  }, []);

  // 2. Health Check (uses relative URL - public endpoint, no auth needed)
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          setStatus(`Online (${data.system || 'Decay_Memory v2.0'})`);
        } else {
          setStatus('Backend Error');
        }
      } catch {
        setStatus('Offline - Check Backend');
      }
    };
    
    checkHealth();
    // Re-check every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await authFetch(`${API_BASE}/chat`, {
        method: 'POST',
        body: JSON.stringify({ message: userMsg.content, project: project }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      
      const aiMsg: Message = { 
        role: 'ai', 
        content: data.response, 
        mood: data.mood,
        intimacy: data.intimacy 
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: `⚠️ Error: Could not connect to the Brain. (${errMsg})` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-area">
          <Cpu className="icon-logo" />
          <h1>Decay_Memory</h1>
        </div>
        <div className="header-center">
          <button 
            className={`view-btn ${view === 'chat' ? 'active' : ''}`}
            onClick={() => setView('chat')}
          >
            <MessageSquare size={16} /> Chat
          </button>
          <button 
            className={`view-btn ${view === 'dashboard' ? 'active' : ''}`}
            onClick={() => setView('dashboard')}
          >
            <Settings size={16} /> Settings
          </button>
        </div>
        <div className="status-bar">
          <span className={`status-indicator ${status.includes('Online') ? 'online' : 'offline'}`}>
            {status}
          </span>
          <div className="project-selector">
            <Terminal size={14} />
            <span>Project: {project}</span>
          </div>
        </div>
      </header>

      {view === 'chat' ? (
        <>
          <main className="chat-window">
            {messages.length === 0 && (
              <div className="empty-state">
                <h2>System Ready.</h2>
                <p>Memory Kernel loaded. Soul Matrix active.</p>
              </div>
            )}
            
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-row ${msg.role}`}>
                <div className="avatar">
                  {msg.role === 'ai' ? <Bot size={20} /> : <User size={20} />}
                </div>
                <div className="message-bubble">
                  <Markdown>{msg.content}</Markdown>
                  {msg.mood && typeof msg.intimacy === 'number' && (
                    <div className="meta-tag">
                      Mood: {msg.mood} • Intimacy: {(msg.intimacy * 100).toFixed(0)}%
                    </div>
                  )}
                  {msg.role === 'ai' && voiceEnabled && (
                    <button 
                      className={`speak-btn ${speakingIndex === idx ? 'speaking' : ''}`}
                      onClick={() => speakMessage(msg.content, idx)}
                      title={speakingIndex === idx ? 'Stop' : 'Speak'}
                    >
                      {speakingIndex === idx ? <VolumeX size={14} /> : <Volume2 size={14} />}
                    </button>
                  )}
                </div>
              </div>
            ))}
            {isLoading && <div className="loading">Thinking...</div>}
            <div ref={messagesEndRef} />
          </main>

          <footer className="input-area">
            <div className="input-wrapper">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Type a message..."
                disabled={isLoading}
              />
              <button onClick={sendMessage} disabled={isLoading}>
                <Send size={18} />
              </button>
            </div>
          </footer>
        </>
      ) : (
        <Dashboard />
      )}
    </div>
  );
}

export default App;
