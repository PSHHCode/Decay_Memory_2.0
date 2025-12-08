"""
Voice Service V1.0 - ElevenLabs Text-to-Speech Integration
Gives the AI a voice, making conversations feel more alive.

Features:
- Text-to-speech via ElevenLabs API
- Voice selection and customization
- Emotional inflection based on Soul state
- Streaming audio support
"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("Voice")

# =============================================================================
# CONFIGURATION
# =============================================================================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Default voice settings
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # "Bella" - warm, friendly female voice
AVAILABLE_VOICES = {
    "bella": "EXAVITQu4vr4xnSDxMaL",      # Warm, friendly
    "rachel": "21m00Tcm4TlvDq8ikWAM",     # Calm, professional
    "domi": "AZnzlk1XvdvUeBnXmlld",       # Strong, confident
    "elli": "MF3mGyEYCl7XYWbV9V6O",       # Young, cheerful
    "josh": "TxGEqnHWrfWFTfGW9XjX",       # Deep, warm male
    "sam": "yoZ06aMxZJJ28mfd3POQ",        # Raspy, authentic male
}

# Voice settings that can be adjusted
DEFAULT_SETTINGS = {
    "stability": 0.5,        # 0-1: Lower = more expressive, Higher = more consistent
    "similarity_boost": 0.75, # 0-1: How closely to match the original voice
    "style": 0.0,            # 0-1: Style exaggeration (v2 voices only)
    "use_speaker_boost": True
}

# Mood to voice settings mapping
MOOD_VOICE_SETTINGS = {
    "joyful": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3},
    "excited": {"stability": 0.3, "similarity_boost": 0.7, "style": 0.4},
    "warm": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2},
    "tired": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.0},
    "sad": {"stability": 0.6, "similarity_boost": 0.9, "style": 0.1},
    "concerned": {"stability": 0.6, "similarity_boost": 0.85, "style": 0.1},
    "playful": {"stability": 0.35, "similarity_boost": 0.7, "style": 0.35},
    "neutral": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
}


# =============================================================================
# ELEVENLABS CLIENT
# =============================================================================

class VoiceService:
    """
    ElevenLabs Text-to-Speech Service
    
    Converts text to speech with emotional awareness.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.current_voice_id = DEFAULT_VOICE_ID
        self.client = None
        
        if self.api_key:
            try:
                from elevenlabs.client import ElevenLabs
                self.client = ElevenLabs(api_key=self.api_key)
                logger.info("🎤 Voice Service initialized with ElevenLabs")
            except ImportError:
                logger.warning("elevenlabs package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs: {e}")
        else:
            logger.warning("No ELEVENLABS_API_KEY set - voice disabled")
    
    def is_available(self) -> bool:
        """Check if voice service is available."""
        return self.client is not None
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the voice by name."""
        voice_id = AVAILABLE_VOICES.get(voice_name.lower())
        if voice_id:
            self.current_voice_id = voice_id
            logger.info(f"Voice set to: {voice_name}")
            return True
        logger.warning(f"Unknown voice: {voice_name}")
        return False
    
    def get_voice_settings(self, mood: str = "neutral") -> Dict[str, Any]:
        """Get voice settings adjusted for current mood."""
        base_settings = DEFAULT_SETTINGS.copy()
        mood_adjustments = MOOD_VOICE_SETTINGS.get(mood, {})
        base_settings.update(mood_adjustments)
        return base_settings
    
    def text_to_speech(
        self, 
        text: str, 
        mood: str = "neutral",
        voice_name: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech audio.
        
        Args:
            text: The text to convert to speech
            mood: Current emotional state (affects voice settings)
            voice_name: Optional voice override
            
        Returns:
            Audio bytes (MP3 format) or None if failed
        """
        if not self.client:
            logger.warning("Voice service not available")
            return None
        
        try:
            from elevenlabs import VoiceSettings
            
            # Get voice ID
            voice_id = self.current_voice_id
            if voice_name:
                voice_id = AVAILABLE_VOICES.get(voice_name.lower(), self.current_voice_id)
            
            # Get mood-adjusted settings
            settings = self.get_voice_settings(mood)
            
            # Generate audio
            audio = self.client.generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=settings["stability"],
                    similarity_boost=settings["similarity_boost"],
                    style=settings.get("style", 0.0),
                    use_speaker_boost=settings.get("use_speaker_boost", True)
                )
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio)
            
            logger.info(f"Generated {len(audio_bytes)} bytes of audio for mood: {mood}")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return None
    
    def text_to_speech_stream(
        self, 
        text: str, 
        mood: str = "neutral",
        voice_name: Optional[str] = None
    ):
        """
        Stream text to speech audio.
        
        Yields audio chunks for real-time playback.
        """
        if not self.client:
            logger.warning("Voice service not available")
            return
        
        try:
            from elevenlabs import VoiceSettings
            
            voice_id = self.current_voice_id
            if voice_name:
                voice_id = AVAILABLE_VOICES.get(voice_name.lower(), self.current_voice_id)
            
            settings = self.get_voice_settings(mood)
            
            # Stream audio
            audio_stream = self.client.generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=settings["stability"],
                    similarity_boost=settings["similarity_boost"],
                    style=settings.get("style", 0.0),
                    use_speaker_boost=settings.get("use_speaker_boost", True)
                ),
                stream=True
            )
            
            for chunk in audio_stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"Text-to-speech stream failed: {e}")
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get list of available voice presets."""
        return AVAILABLE_VOICES.copy()
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get ElevenLabs account info (quota, etc.)."""
        if not self.client:
            return None
        
        try:
            user = self.client.user.get()
            subscription = user.subscription
            return {
                "character_count": subscription.character_count,
                "character_limit": subscription.character_limit,
                "remaining": subscription.character_limit - subscription.character_count,
                "tier": subscription.tier
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None


# =============================================================================
# SPEECH-TO-TEXT (WHISPER)
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SpeechToText:
    """
    OpenAI Whisper Speech-to-Text Service
    
    Converts audio to text for voice input.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("🎧 Speech-to-Text initialized with Whisper")
            except ImportError:
                logger.warning("openai package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Whisper: {e}")
        else:
            logger.warning("No OPENAI_API_KEY set - speech-to-text disabled")
    
    def is_available(self) -> bool:
        """Check if speech-to-text is available."""
        return self.client is not None
    
    def transcribe(self, audio_file, language: str = "en") -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_file: File-like object containing audio (wav, mp3, etc.)
            language: ISO language code (default: English)
            
        Returns:
            Transcribed text or None if failed
        """
        if not self.client:
            logger.warning("Speech-to-text not available")
            return None
        
        try:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
            logger.info(f"Transcribed audio: {len(transcription.text)} chars")
            return transcription.text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav", language: str = "en") -> Optional[str]:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Filename with extension for format detection
            language: ISO language code
            
        Returns:
            Transcribed text or None
        """
        if not self.client:
            return None
        
        try:
            import io
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename
            return self.transcribe(audio_file, language)
        except Exception as e:
            logger.error(f"Transcription from bytes failed: {e}")
            return None


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_voice_service: Optional[VoiceService] = None
_stt_service: Optional[SpeechToText] = None

def get_voice_service() -> VoiceService:
    """Get or create the voice service singleton."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service

def get_stt_service() -> SpeechToText:
    """Get or create the speech-to-text service singleton."""
    global _stt_service
    if _stt_service is None:
        _stt_service = SpeechToText()
    return _stt_service

