import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Terminal, Cpu, Settings, MessageSquare, MicOff, Phone, PhoneOff } from 'lucide-react';
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
  const [voiceChatMode, setVoiceChatMode] = useState(false);
  const [voiceChatStatus, setVoiceChatStatus] = useState<'idle' | 'listening' | 'thinking' | 'speaking'>('idle');
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const exitingVoiceModeRef = useRef(false);  // Track when we're exiting to avoid phantom transcriptions
  
  // ===== VOICE CHAT MODE ("Her" style) =====
  
  const startVoiceChatMode = async () => {
    try {
      // Request microphone permission upfront
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setVoiceChatMode(true);
      setVoiceChatStatus('listening');
      startVoiceChatListening(stream);
    } catch (error) {
      console.error('Microphone access denied:', error);
      alert('Voice chat requires microphone permission. Please allow access and try again.');
    }
  };
  
  const stopVoiceChatMode = () => {
    // Set flag BEFORE stopping recorder to prevent phantom transcription
    exitingVoiceModeRef.current = true;
    
    setVoiceChatMode(false);
    setVoiceChatStatus('idle');
    
    // Stop any recording (onstop will check exitingVoiceModeRef)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    // Stop audio playback
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    
    // Release microphone
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Reset flag after a short delay
    setTimeout(() => { exitingVoiceModeRef.current = false; }, 100);
  };
  
  const startVoiceChatListening = (stream: MediaStream) => {
    console.log('Starting voice chat listening...');
    
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];
    
    mediaRecorder.ondataavailable = (event) => {
      console.log('Audio data received:', event.data.size, 'bytes');
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };
    
    mediaRecorder.onstop = async () => {
      console.log('MediaRecorder stopped, processing audio...');
      
      // Check if we're exiting voice mode - if so, don't process audio
      if (exitingVoiceModeRef.current) {
        console.log('Exiting voice mode, skipping transcription');
        return;
      }
      
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      console.log('Audio blob size:', audioBlob.size);
      
      // Only process if there's actual audio (more than ~1KB)
      if (audioBlob.size > 1000) {
        await processVoiceChatInput(audioBlob);
      } else {
        console.log('Audio too short, restarting listening...');
        // Restart listening if audio was too short
        if (streamRef.current) {
          startVoiceChatListening(streamRef.current);
        }
      }
    };
    
    // Start recording with timeslice to collect data periodically
    mediaRecorder.start(1000); // Collect data every 1 second
    setVoiceChatStatus('listening');
    console.log('MediaRecorder started, state:', mediaRecorder.state);
  };
  
  const stopVoiceChatListening = () => {
    console.log('stopVoiceChatListening called');
    console.log('mediaRecorderRef.current:', mediaRecorderRef.current);
    console.log('state:', mediaRecorderRef.current?.state);
    
    if (mediaRecorderRef.current) {
      const state = mediaRecorderRef.current.state;
      if (state === 'recording' || state === 'paused') {
        console.log('Stopping MediaRecorder...');
        mediaRecorderRef.current.stop();
        setVoiceChatStatus('thinking');
      } else {
        console.log('MediaRecorder not in recording state:', state);
      }
    } else {
      console.log('No mediaRecorder ref!');
    }
  };
  
  const processVoiceChatInput = async (audioBlob: Blob) => {
    setVoiceChatStatus('thinking');
    
    try {
      // 1. Transcribe
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      
      const headers: HeadersInit = {};
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      
      const transcribeRes = await fetch(`${API_BASE}/voice/transcribe`, {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!transcribeRes.ok) {
        throw new Error('Transcription failed');
      }
      
      const { text } = await transcribeRes.json();
      
      if (!text || !text.trim()) {
        // No speech detected, restart listening
        if (voiceChatMode && streamRef.current) {
          startVoiceChatListening(streamRef.current);
        }
        return;
      }
      
      // Add user message to chat
      const userMsg: Message = { role: 'user', content: text };
      setMessages(prev => [...prev, userMsg]);
      
      // 2. Get AI response
      const chatRes = await authFetch(`${API_BASE}/chat`, {
        method: 'POST',
        body: JSON.stringify({ message: text, project: project }),
      });
      
      if (!chatRes.ok) {
        throw new Error('Chat failed');
      }
      
      const chatData = await chatRes.json();
      
      // Add AI message to chat
      const aiMsg: Message = { 
        role: 'ai', 
        content: chatData.response, 
        mood: chatData.mood,
        intimacy: chatData.intimacy 
      };
      setMessages(prev => [...prev, aiMsg]);
      
      // 3. Speak the response
      console.log('Speaking response...');
      setVoiceChatStatus('speaking');
      await speakAndWait(chatData.response);
      console.log('Speech finished!');
      
      // 4. Restart listening (continuous conversation)
      console.log('Checking for restart - voiceChatMode:', voiceChatMode, 'streamRef:', streamRef.current);
      if (streamRef.current) {
        console.log('Restarting listening...');
        setVoiceChatStatus('listening');
        startVoiceChatListening(streamRef.current);
      } else {
        console.log('No stream ref, cannot restart listening');
        setVoiceChatStatus('idle');
      }
      
    } catch (error) {
      console.error('Voice chat error:', error);
      setVoiceChatStatus('idle');
      // Restart listening on error
      if (streamRef.current) {
        setTimeout(() => {
          if (streamRef.current) {
            console.log('Restarting after error...');
            setVoiceChatStatus('listening');
            startVoiceChatListening(streamRef.current);
          }
        }, 1000);
      }
    }
  };
  
  const speakAndWait = (text: string): Promise<void> => {
    return new Promise(async (resolve) => {
      console.log('speakAndWait: starting for text length:', text.length);
      try {
        const response = await authFetch(`${API_BASE}/voice/speak`, {
          method: 'POST',
          body: JSON.stringify({ message: text }),
        });
        
        console.log('speakAndWait: got response, ok:', response.ok);
        
        if (response.ok) {
          const audioBlob = await response.blob();
          console.log('speakAndWait: audio blob size:', audioBlob.size);
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          audioRef.current = audio;
          
          audio.onended = () => {
            console.log('speakAndWait: audio ended naturally');
            URL.revokeObjectURL(audioUrl);
            audioRef.current = null;
            resolve();
          };
          
          audio.onerror = (e) => {
            console.log('speakAndWait: audio error:', e);
            URL.revokeObjectURL(audioUrl);
            audioRef.current = null;
            resolve();
          };
          
          console.log('speakAndWait: playing audio...');
          audio.play().catch(e => {
            console.log('speakAndWait: play() error:', e);
            resolve();
          });
        } else {
          console.log('speakAndWait: response not ok');
          resolve();
        }
      } catch (err) {
        console.log('speakAndWait: catch error:', err);
        resolve();
      }
    });
  };
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  
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

  
  // Send message with optional text override (for voice input)
  const sendMessageWithText = async (text: string) => {
    if (!text.trim()) return;
    
    const userMsg: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await authFetch(`${API_BASE}/chat`, {
        method: 'POST',
        body: JSON.stringify({ message: text, project: project }),
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

  const sendMessage = async () => {
    if (!input.trim()) return;
    await sendMessageWithText(input);
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
            type="button"
            className={`view-btn ${view === 'chat' ? 'active' : ''}`}
            onClick={() => { console.log('Chat clicked'); setView('chat'); }}
          >
            <MessageSquare size={16} /> Chat
          </button>
          <button 
            type="button"
            className={`view-btn ${view === 'dashboard' ? 'active' : ''}`}
            onClick={() => { console.log('Settings clicked'); setView('dashboard'); }}
          >
            <Settings size={16} /> Settings
          </button>
        </div>
        <div className="status-bar">
          <button 
            className="voice-chat-btn"
            onClick={startVoiceChatMode}
            title="Start Voice Chat"
          >
            <Phone size={14} /> Voice
          </button>
          <span className={`status-indicator ${status.includes('Online') ? 'online' : 'offline'}`}>
            {status}
          </span>
          <div className="project-selector">
            <Terminal size={14} />
            <span>Project: {project}</span>
          </div>
        </div>
      </header>

      {/* Voice Chat Mode Overlay */}
      {voiceChatMode && (
        <div className="voice-chat-overlay">
          <div className="voice-chat-container">
            <div className={`voice-orb ${voiceChatStatus}`}>
              <div className="orb-inner"></div>
            </div>
            <div className="voice-status-text">
              {voiceChatStatus === 'listening' && 'Listening...'}
              {voiceChatStatus === 'thinking' && 'Thinking...'}
              {voiceChatStatus === 'speaking' && 'Speaking...'}
              {voiceChatStatus === 'idle' && 'Ready'}
            </div>
            <div className="voice-chat-controls">
              {voiceChatStatus === 'listening' && (
                <button 
                  type="button"
                  className="voice-stop-btn" 
                  onClick={() => { console.log('Done Speaking clicked!'); stopVoiceChatListening(); }}
                >
                  <MicOff size={20} /> Done Speaking
                </button>
              )}
              <button 
                type="button"
                className="voice-exit-btn" 
                onClick={() => { console.log('End Voice Chat clicked!'); stopVoiceChatMode(); }}
              >
                <PhoneOff size={20} /> End Voice Chat
              </button>
            </div>
            <p className="voice-hint">
              {voiceChatStatus === 'listening' && 'Speak naturally, then click "Done Speaking" when finished'}
              {voiceChatStatus === 'thinking' && 'Processing your message...'}
              {voiceChatStatus === 'speaking' && 'AI is responding...'}
            </p>
          </div>
        </div>
      )}

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
