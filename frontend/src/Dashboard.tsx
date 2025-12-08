import { useState, useEffect } from 'react';
import { 
  Settings, Sliders, Heart, Brain, Volume2, Clock, 
  Sparkles, RefreshCw, Save, AlertCircle, Check
} from 'lucide-react';
import './Dashboard.css';

// =============================================================================
// Types
// =============================================================================

interface SoulState {
  mood: string;
  energy: number;
  intimacy: number;
  vad: { valence: number; arousal: number; dominance: number };
  circadian_phase: string;
}

interface VoiceStatus {
  available: boolean;
  current_voice: string;
  account: { remaining: number; character_limit: number } | null;
}

interface DashboardConfig {
  half_lives: Record<string, number>;
  proactive_retrieval: Record<string, number | boolean>;
  conflict_resolution: Record<string, number>;
  knowledge_graph: Record<string, number | boolean>;
  feedback_system: Record<string, number>;
  session_management: Record<string, number>;
}

interface HeartbeatStatus {
  running: boolean;
  interval_seconds: number;
  quiet_hours: { start: number; end: number };
  last_beat: string | null;
}

interface CuriosityStatus {
  running: boolean;
  queue_size: number;
  explored_today: number;
  max_daily: number;
}

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const API_KEY = import.meta.env.VITE_API_KEY || '';

const authFetch = async (url: string, options: RequestInit = {}) => {
  const headers = new Headers(options.headers);
  if (API_KEY) headers.set('X-API-Key', API_KEY);
  headers.set('Content-Type', 'application/json');
  return fetch(url, { ...options, headers });
};

// =============================================================================
// Voice Options
// =============================================================================

const VOICE_OPTIONS = [
  { id: 'bella', name: 'Bella', description: 'Warm, friendly' },
  { id: 'rachel', name: 'Rachel', description: 'Calm, professional' },
  { id: 'domi', name: 'Domi', description: 'Strong, confident' },
  { id: 'elli', name: 'Elli', description: 'Young, cheerful' },
  { id: 'josh', name: 'Josh', description: 'Deep, warm male' },
  { id: 'sam', name: 'Sam', description: 'Raspy, authentic male' },
];

// =============================================================================
// Dashboard Component
// =============================================================================

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'user' | 'advanced'>('user');
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  
  // User Settings State
  const [soul, setSoul] = useState<SoulState | null>(null);
  const [voice, setVoice] = useState<VoiceStatus | null>(null);
  const [selectedVoice, setSelectedVoice] = useState('bella');
  const [voiceStability, setVoiceStability] = useState(0.5);
  const [intimacyOverride, setIntimacyOverride] = useState<number | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [testingVoice, setTestingVoice] = useState(false);
  
  // Advanced Settings State
  const [config, setConfig] = useState<DashboardConfig | null>(null);
  const [heartbeat, setHeartbeat] = useState<HeartbeatStatus | null>(null);
  const [curiosity, setCuriosity] = useState<CuriosityStatus | null>(null);
  
  // =============================================================================
  // Data Loading
  // =============================================================================
  
  useEffect(() => {
    loadAllData();
  }, []);
  
  const loadAllData = async () => {
    try {
      // Load Soul State
      const soulRes = await authFetch(`${API_BASE}/soul/state`);
      if (soulRes.ok) setSoul(await soulRes.json());
      
      // Load Voice Status
      const voiceRes = await authFetch(`${API_BASE}/voice/status`);
      if (voiceRes.ok) setVoice(await voiceRes.json());
      
      // Load Config
      const configRes = await authFetch(`${API_BASE}/config`);
      if (configRes.ok) setConfig(await configRes.json());
      
      // Load Heartbeat Status
      const hbRes = await authFetch(`${API_BASE}/heartbeat/status`);
      if (hbRes.ok) setHeartbeat(await hbRes.json());
      
      // Load Curiosity Status
      const curRes = await authFetch(`${API_BASE}/curiosity/status`);
      if (curRes.ok) setCuriosity(await curRes.json());
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };
  
  // =============================================================================
  // Save Handlers
  // =============================================================================
  
  const saveUserSettings = async () => {
    setSaving(true);
    setSaveStatus('idle');
    
    try {
      // Update voice
      if (selectedVoice) {
        await authFetch(`${API_BASE}/voice/set/${selectedVoice}`, { method: 'POST' });
      }
      
      // Update voice settings (stability/expressiveness)
      await authFetch(`${API_BASE}/voice/settings`, {
        method: 'PUT',
        body: JSON.stringify({ stability: voiceStability }),
      });
      
      // Update soul state (intimacy, energy, mood)
      if (intimacyOverride !== null || selectedMood) {
        const soulUpdate: Record<string, unknown> = {};
        if (intimacyOverride !== null) soulUpdate.intimacy = intimacyOverride;
        if (selectedMood) soulUpdate.mood = selectedMood;
        
        await authFetch(`${API_BASE}/soul/state`, {
          method: 'PUT',
          body: JSON.stringify(soulUpdate),
        });
      }
      
      setSaveStatus('success');
      await loadAllData();
    } catch {
      setSaveStatus('error');
    } finally {
      setSaving(false);
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };
  
  const saveAdvancedSettings = async () => {
    setSaving(true);
    setSaveStatus('idle');
    
    try {
      if (config) {
        await authFetch(`${API_BASE}/config`, {
          method: 'PUT',
          body: JSON.stringify(config),
        });
      }
      
      setSaveStatus('success');
      await loadAllData();
    } catch {
      setSaveStatus('error');
    } finally {
      setSaving(false);
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };
  
  // =============================================================================
  // Helper: Update nested config value
  // =============================================================================
  
  const updateConfig = (section: string, key: string, value: number | boolean) => {
    if (!config) return;
    setConfig({
      ...config,
      [section]: {
        ...config[section as keyof DashboardConfig],
        [key]: value,
      },
    });
  };
  
  // =============================================================================
  // Render
  // =============================================================================
  
  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h2><Settings size={24} /> Control Panel</h2>
        <div className="tab-switcher">
          <button 
            className={activeTab === 'user' ? 'active' : ''} 
            onClick={() => setActiveTab('user')}
          >
            <Heart size={16} /> User Settings
          </button>
          <button 
            className={activeTab === 'advanced' ? 'active' : ''} 
            onClick={() => setActiveTab('advanced')}
          >
            <Sliders size={16} /> Advanced
          </button>
        </div>
      </div>
      
      {/* Content */}
      <div className="dashboard-content">
        {activeTab === 'user' ? (
          <UserSettingsTab 
            soul={soul}
            voice={voice}
            selectedVoice={selectedVoice}
            setSelectedVoice={setSelectedVoice}
            voiceStability={voiceStability}
            setVoiceStability={setVoiceStability}
            intimacyOverride={intimacyOverride}
            setIntimacyOverride={setIntimacyOverride}
            selectedMood={selectedMood}
            setSelectedMood={setSelectedMood}
            testingVoice={testingVoice}
            setTestingVoice={setTestingVoice}
            apiBase={API_BASE}
            apiKey={API_KEY}
          />
        ) : (
          <AdvancedSettingsTab 
            config={config}
            updateConfig={updateConfig}
            heartbeat={heartbeat}
            curiosity={curiosity}
          />
        )}
      </div>
      
      {/* Footer with Save */}
      <div className="dashboard-footer">
        <button 
          className="refresh-btn" 
          onClick={loadAllData}
          title="Refresh data"
        >
          <RefreshCw size={16} />
        </button>
        
        <div className="save-area">
          {saveStatus === 'success' && (
            <span className="save-status success"><Check size={14} /> Saved!</span>
          )}
          {saveStatus === 'error' && (
            <span className="save-status error"><AlertCircle size={14} /> Error saving</span>
          )}
          
          <button 
            className="save-btn"
            onClick={activeTab === 'user' ? saveUserSettings : saveAdvancedSettings}
            disabled={saving}
          >
            <Save size={16} /> {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// User Settings Tab
// =============================================================================

// Available moods for manual selection
const MOOD_OPTIONS = [
  { id: 'neutral', label: 'Neutral', emoji: '😐' },
  { id: 'warm', label: 'Warm', emoji: '🥰' },
  { id: 'joyful', label: 'Joyful', emoji: '😊' },
  { id: 'playful', label: 'Playful', emoji: '😜' },
  { id: 'curious', label: 'Curious', emoji: '🤔' },
  { id: 'thoughtful', label: 'Thoughtful', emoji: '💭' },
  { id: 'concerned', label: 'Concerned', emoji: '😟' },
  { id: 'tired', label: 'Tired', emoji: '😴' },
  { id: 'calm', label: 'Calm', emoji: '😌' },
];

interface UserSettingsProps {
  soul: SoulState | null;
  voice: VoiceStatus | null;
  selectedVoice: string;
  setSelectedVoice: (v: string) => void;
  voiceStability: number;
  setVoiceStability: (v: number) => void;
  intimacyOverride: number | null;
  setIntimacyOverride: (v: number | null) => void;
  selectedMood: string | null;
  setSelectedMood: (m: string | null) => void;
  testingVoice: boolean;
  setTestingVoice: (t: boolean) => void;
  apiBase: string;
  apiKey: string;
}

function UserSettingsTab({
  soul, voice, selectedVoice, setSelectedVoice,
  voiceStability, setVoiceStability, intimacyOverride, setIntimacyOverride,
  selectedMood, setSelectedMood, testingVoice, setTestingVoice, apiBase, apiKey
}: UserSettingsProps) {
  
  const displayIntimacy = intimacyOverride ?? soul?.intimacy ?? 0.5;
  const displayMood = selectedMood ?? soul?.mood ?? 'neutral';
  
  // Test voice function
  const testVoice = async () => {
    setTestingVoice(true);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;
      
      const response = await fetch(`${apiBase}/voice/test`, {
        method: 'POST',
        headers,
      });
      
      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
        audio.onended = () => URL.revokeObjectURL(audioUrl);
      }
    } catch (error) {
      console.error('Voice test failed:', error);
    } finally {
      setTestingVoice(false);
    }
  };
  
  return (
    <div className="settings-grid">
      {/* Soul Status Card */}
      <div className="settings-card">
        <h3><Brain size={18} /> AI Emotional State</h3>
        {soul ? (
          <div className="soul-display">
            <div className="soul-stat">
              <label>Current Mood</label>
              <span className="mood-badge">{soul.mood}</span>
            </div>
            <div className="soul-stat">
              <label>Energy</label>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${soul.energy * 100}%` }} />
              </div>
              <span>{(soul.energy * 100).toFixed(0)}%</span>
            </div>
            <div className="soul-stat">
              <label>Circadian Phase</label>
              <span>{soul.circadian_phase}</span>
            </div>
            <div className="vad-display">
              <span title="Valence (positive/negative)">V: {soul.vad.valence.toFixed(2)}</span>
              <span title="Arousal (calm/excited)">A: {soul.vad.arousal.toFixed(2)}</span>
              <span title="Dominance (submissive/dominant)">D: {soul.vad.dominance.toFixed(2)}</span>
            </div>
          </div>
        ) : (
          <p className="loading-text">Loading soul state...</p>
        )}
      </div>
      
      {/* Mood Selector */}
      <div className="settings-card">
        <h3><Sparkles size={18} /> Set AI Mood</h3>
        <p className="card-description">
          Override the AI's current mood. This affects personality and responses.
        </p>
        <div className="mood-grid">
          {MOOD_OPTIONS.map((m) => (
            <button
              key={m.id}
              className={`mood-option ${displayMood === m.id ? 'selected' : ''}`}
              onClick={() => setSelectedMood(m.id)}
            >
              <span className="mood-emoji">{m.emoji}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </div>
      </div>
      
      {/* Intimacy Slider */}
      <div className="settings-card">
        <h3><Heart size={18} /> Relationship Level</h3>
        <p className="card-description">
          Adjust how close the AI feels to you. Higher = more personal, warmer tone.
        </p>
        <div className="slider-control">
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.05"
            value={displayIntimacy}
            onChange={(e) => setIntimacyOverride(parseFloat(e.target.value))}
          />
          <span className="slider-value">{(displayIntimacy * 100).toFixed(0)}%</span>
        </div>
        <div className="slider-labels">
          <span>Formal</span>
          <span>Friendly</span>
          <span>Close</span>
        </div>
      </div>
      
      {/* Voice Settings */}
      <div className="settings-card wide">
        <h3><Volume2 size={18} /> Voice Settings</h3>
        {voice?.available ? (
          <>
            <div className="voice-grid">
              {VOICE_OPTIONS.map((v) => (
                <button
                  key={v.id}
                  className={`voice-option ${selectedVoice === v.id ? 'selected' : ''}`}
                  onClick={() => setSelectedVoice(v.id)}
                >
                  <strong>{v.name}</strong>
                  <span>{v.description}</span>
                </button>
              ))}
            </div>
            
            <div className="voice-stability">
              <label>Expressiveness</label>
              <p className="card-description">
                Lower = more emotional variation. Higher = more consistent/stable.
              </p>
              <div className="slider-control">
                <input 
                  type="range" 
                  min="0.2" 
                  max="0.8" 
                  step="0.05"
                  value={voiceStability}
                  onChange={(e) => setVoiceStability(parseFloat(e.target.value))}
                />
                <span className="slider-value">{voiceStability.toFixed(2)}</span>
              </div>
              <div className="slider-labels">
                <span>Expressive</span>
                <span>Balanced</span>
                <span>Stable</span>
              </div>
            </div>
            
            <div className="voice-actions">
              <button 
                className="test-voice-btn"
                onClick={testVoice}
                disabled={testingVoice}
              >
                <Volume2 size={16} /> {testingVoice ? 'Playing...' : 'Test Voice'}
              </button>
            </div>
            
            {voice.account && (
              <div className="voice-quota">
                <span>Characters remaining: {voice.account.remaining?.toLocaleString() ?? 'Unknown'} / {voice.account.character_limit?.toLocaleString() ?? '?'}</span>
              </div>
            )}
          </>
        ) : (
          <p className="disabled-notice">
            <AlertCircle size={16} /> Voice service not available. Add ELEVENLABS_API_KEY to enable.
          </p>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Advanced Settings Tab
// =============================================================================

interface AdvancedSettingsProps {
  config: DashboardConfig | null;
  updateConfig: (section: string, key: string, value: number | boolean) => void;
  heartbeat: HeartbeatStatus | null;
  curiosity: CuriosityStatus | null;
}

function AdvancedSettingsTab({ config, updateConfig, heartbeat, curiosity }: AdvancedSettingsProps) {
  if (!config) {
    return <p className="loading-text">Loading configuration...</p>;
  }
  
  return (
    <div className="settings-grid">
      {/* Memory Decay */}
      <div className="settings-card">
        <h3><Clock size={18} /> Memory Decay Rates</h3>
        <p className="card-description">
          Half-life in days. Higher = memories last longer.
        </p>
        <div className="config-list">
          {Object.entries(config.half_lives)
            .filter(([k]) => !k.startsWith('_'))
            .map(([key, value]) => (
              <div key={key} className="config-item">
                <label>{key.replace(/_/g, ' ')}</label>
                <input 
                  type="number"
                  value={Math.round(Number(value) / 86400)} // Convert seconds to days
                  onChange={(e) => updateConfig('half_lives', key, parseInt(e.target.value) * 86400)}
                  min={1}
                  max={730}
                />
                <span className="unit">days</span>
              </div>
            ))}
        </div>
      </div>
      
      {/* Proactive Retrieval */}
      <div className="settings-card">
        <h3><Sparkles size={18} /> Proactive Retrieval</h3>
        <p className="card-description">
          Controls when AI automatically surfaces memories.
        </p>
        <div className="config-list">
          <div className="config-item toggle">
            <label>Enabled</label>
            <input 
              type="checkbox"
              checked={config.proactive_retrieval.enabled as boolean}
              onChange={(e) => updateConfig('proactive_retrieval', 'enabled', e.target.checked)}
            />
          </div>
          <div className="config-item">
            <label>Trigger Threshold</label>
            <input 
              type="number"
              value={config.proactive_retrieval.trigger_threshold as number}
              onChange={(e) => updateConfig('proactive_retrieval', 'trigger_threshold', parseFloat(e.target.value))}
              min={0.1}
              max={0.95}
              step={0.05}
            />
          </div>
          <div className="config-item">
            <label>Injection Threshold</label>
            <input 
              type="number"
              value={config.proactive_retrieval.injection_threshold as number}
              onChange={(e) => updateConfig('proactive_retrieval', 'injection_threshold', parseFloat(e.target.value))}
              min={0.1}
              max={0.95}
              step={0.05}
            />
          </div>
          <div className="config-item">
            <label>Max Injections</label>
            <input 
              type="number"
              value={config.proactive_retrieval.max_injections as number}
              onChange={(e) => updateConfig('proactive_retrieval', 'max_injections', parseInt(e.target.value))}
              min={1}
              max={10}
            />
          </div>
        </div>
      </div>
      
      {/* Conflict Resolution */}
      <div className="settings-card">
        <h3><AlertCircle size={18} /> Conflict Resolution</h3>
        <p className="card-description">
          Thresholds for detecting contradictory memories.
        </p>
        <div className="config-list">
          <div className="config-item">
            <label>Similarity Threshold</label>
            <input 
              type="number"
              value={config.conflict_resolution.similarity_threshold}
              onChange={(e) => updateConfig('conflict_resolution', 'similarity_threshold', parseFloat(e.target.value))}
              min={0.5}
              max={0.99}
              step={0.05}
            />
          </div>
          <div className="config-item">
            <label>Confidence Threshold</label>
            <input 
              type="number"
              value={config.conflict_resolution.confidence_threshold}
              onChange={(e) => updateConfig('conflict_resolution', 'confidence_threshold', parseFloat(e.target.value))}
              min={0.5}
              max={0.99}
              step={0.05}
            />
          </div>
        </div>
      </div>
      
      {/* Feedback System */}
      <div className="settings-card">
        <h3><Sliders size={18} /> Memory Feedback</h3>
        <p className="card-description">
          How boost/deprecate commands affect memory importance.
        </p>
        <div className="config-list">
          <div className="config-item">
            <label>Boost Increment</label>
            <input 
              type="number"
              value={config.feedback_system.boost_increment}
              onChange={(e) => updateConfig('feedback_system', 'boost_increment', parseFloat(e.target.value))}
              min={1.1}
              max={3.0}
              step={0.1}
            />
          </div>
          <div className="config-item">
            <label>Deprecation Multiplier</label>
            <input 
              type="number"
              value={config.feedback_system.deprecation_multiplier}
              onChange={(e) => updateConfig('feedback_system', 'deprecation_multiplier', parseFloat(e.target.value))}
              min={0.01}
              max={0.5}
              step={0.01}
            />
          </div>
          <div className="config-item">
            <label>Max Boost</label>
            <input 
              type="number"
              value={config.feedback_system.max_boost}
              onChange={(e) => updateConfig('feedback_system', 'max_boost', parseFloat(e.target.value))}
              min={2.0}
              max={10.0}
              step={0.5}
            />
          </div>
        </div>
      </div>
      
      {/* Heartbeat Status */}
      <div className="settings-card">
        <h3><Heart size={18} /> Heartbeat Service</h3>
        {heartbeat ? (
          <div className="status-display">
            <div className="status-row">
              <span>Status</span>
              <span className={`status-badge ${heartbeat.running ? 'running' : 'stopped'}`}>
                {heartbeat.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="status-row">
              <span>Interval</span>
              <span>{heartbeat.interval_seconds}s ({Math.round(heartbeat.interval_seconds / 60)} min)</span>
            </div>
            <div className="status-row">
              <span>Quiet Hours</span>
              <span>{heartbeat.quiet_hours.start}:00 - {heartbeat.quiet_hours.end}:00</span>
            </div>
          </div>
        ) : (
          <p className="loading-text">Loading...</p>
        )}
      </div>
      
      {/* Curiosity Engine Status */}
      <div className="settings-card">
        <h3><Sparkles size={18} /> Curiosity Engine</h3>
        {curiosity ? (
          <div className="status-display">
            <div className="status-row">
              <span>Status</span>
              <span className={`status-badge ${curiosity.running ? 'running' : 'stopped'}`}>
                {curiosity.running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="status-row">
              <span>Queue Size</span>
              <span>{curiosity.queue_size} topics</span>
            </div>
            <div className="status-row">
              <span>Explored Today</span>
              <span>{curiosity.explored_today} / {curiosity.max_daily}</span>
            </div>
          </div>
        ) : (
          <p className="loading-text">Loading...</p>
        )}
      </div>
      
      {/* Knowledge Graph */}
      <div className="settings-card">
        <h3><Brain size={18} /> Knowledge Graph</h3>
        <div className="config-list">
          <div className="config-item toggle">
            <label>Auto Invalidate</label>
            <input 
              type="checkbox"
              checked={config.knowledge_graph.auto_invalidate as boolean}
              onChange={(e) => updateConfig('knowledge_graph', 'auto_invalidate', e.target.checked)}
            />
          </div>
          <div className="config-item">
            <label>Decay Days</label>
            <input 
              type="number"
              value={config.knowledge_graph.decay_days as number}
              onChange={(e) => updateConfig('knowledge_graph', 'decay_days', parseInt(e.target.value))}
              min={7}
              max={730}
            />
            <span className="unit">days</span>
          </div>
          <div className="config-item">
            <label>Cache Duration</label>
            <input 
              type="number"
              value={config.knowledge_graph.cache_seconds as number}
              onChange={(e) => updateConfig('knowledge_graph', 'cache_seconds', parseInt(e.target.value))}
              min={60}
              max={3600}
            />
            <span className="unit">sec</span>
          </div>
        </div>
      </div>
    </div>
  );
}

