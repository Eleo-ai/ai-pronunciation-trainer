"""
OpenAI Whisper API Model for Serverless Deployment
Uses OpenAI's Whisper API for speech recognition
"""

import numpy as np
from ModelInterfaces import IASRModel
from typing import Union
import os
import requests
import tempfile
import soundfile as sf


class WhisperAPIModel(IASRModel):
    """
    OpenAI Whisper API implementation for serverless environments.
    
    Environment variables required:
    - WHISPER_API_KEY: Your OpenAI API key
    - OPENAI_API_BASE: (Optional) Custom API base URL, defaults to https://api.openai.com/v1
    
    Get your API key at: https://platform.openai.com/api-keys
    """
    
    def __init__(self, api_key=None, api_base=None):
        self.api_key = api_key or os.getenv('WHISPER_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.api_base = api_base or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError(
                "WHISPER_API_KEY or OPENAI_API_KEY environment variable must be set. "
                "Get your key at: https://platform.openai.com/api-keys"
            )
        
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000
        self.endpoint = f"{self.api_base.rstrip('/')}/audio/transcriptions"
    
    def processAudio(self, audio: Union[np.ndarray, list]):
        """
        Process audio through OpenAI Whisper API.
        
        Args:
            audio: numpy array or list with shape (1, samples) or (samples,)
        """
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Ensure proper shape
        if audio.ndim == 2:
            audio = audio[0]  # Take first channel
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)
        
        try:
            # Call OpenAI Whisper API
            result = self._call_openai_api(tmp_path)
            self._transcript = result['text']
            self._word_locations = result['word_locations']
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _call_openai_api(self, audio_path):
        """Call OpenAI Whisper API"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file, 'audio/wav')
            }
            data = {
                'model': 'whisper-large-v3-turbo',
                'response_format': 'verbose_json',
                'timestamp_granularities[]': 'word'
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(
                f"OpenAI API Error: {response.status_code} - {response.text}\n"
                f"Make sure your API key is valid and has credits available."
            )
        
        result = response.json()
        
        # Parse response
        text = result.get('text', '')
        words = result.get('words', [])
        
        word_locations = []
        for word_info in words:
            word_locations.append({
                'word': word_info.get('word', '').strip(),
                'start_ts': word_info.get('start', 0) * self.sample_rate,
                'end_ts': word_info.get('end', 0) * self.sample_rate,
                'tag': 'processed'
            })
        
        return {
            'text': text,
            'word_locations': word_locations
        }
    
    def getTranscript(self) -> str:
        """Get the transcript from the processed audio"""
        return self._transcript
    
    def getWordLocations(self) -> list:
        """Get word timestamps from the processed audio"""
        return self._word_locations


# Convenience function for backward compatibility
def get_api_asr_model(api_key=None):
    """
    Factory function to create OpenAI Whisper API model.
    
    Usage:
        # Using environment variables
        model = get_api_asr_model()
        
        # Or specify explicitly
        model = get_api_asr_model(api_key='sk-...')
    """
    return WhisperAPIModel(api_key=api_key)

