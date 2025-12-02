"""
AWS SageMaker Whisper Model Wrapper for Serverless Deployment
Uses AWS SageMaker deployed Whisper model for speech recognition
"""

import numpy as np
from ModelInterfaces import IASRModel
from typing import Union
import os
import boto3
import json
import tempfile
import soundfile as sf
import base64


class WhisperAPIModel(IASRModel):
    """
    AWS SageMaker Whisper implementation for serverless environments.
    
    Environment variables optional:
    - SAGEMAKER_ENDPOINT_NAME: SageMaker endpoint name (defaults to 'redparrot-whisper-base-provisioned')
    - AWS_REGION: AWS region for SageMaker (defaults to 'eu-central-1')
    
    AWS credentials should be configured via IAM role or environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    
    Args:
        endpoint_name: SageMaker endpoint name (optional)
        region_name: AWS region (optional)
        language: ISO 639-1 language code (e.g., 'en', 'pl', 'de') for improved accuracy (optional)
    """
    
    def __init__(self, endpoint_name=None, region_name=None, language=None):
        self.endpoint_name = endpoint_name or os.getenv('SAGEMAKER_ENDPOINT_NAME', 'redparrot-whisper-base-provisioned')
        self.region_name = region_name or os.getenv('AWS_REGION', 'eu-central-1')
        self.language = language  # ISO 639-1 language code (e.g., 'en', 'pl', 'de')
        
        # Initialize SageMaker runtime client
        self.sagemaker_runtime = boto3.client(
            'sagemaker-runtime',
            region_name=self.region_name
        )
        
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000
    
    def processAudio(self, audio: Union[np.ndarray, list]):
        """
        Process audio through AWS SageMaker Whisper endpoint.
        
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
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        #     tmp_path = tmp_file.name
        #     sf.write(tmp_path, audio, self.sample_rate)
        
        try:
            # Call SageMaker Whisper endpoint
            result = self._call_sagemaker_endpoint(audio)
            self._transcript = result['text']
            self._word_locations = result['word_locations']
            
        finally:
            # Clean up temp file
            # if os.path.exists(tmp_path):
            #     os.remove(tmp_path)
            pass
    
    def _call_sagemaker_endpoint(self, audio):
        """Call AWS SageMaker Whisper endpoint"""
        
        # Read audio file and encode as base64
        # with open(audio_path, 'rb') as audio_file:
        #     audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        payload = {
            "audio": audio.tolist(),  # Convert numpy array to JSON-serializable list
            "timestamps": True
        }
        
        # Add language if specified (Whisper supports ISO 639-1 codes like 'en', 'pl', 'de')
        if self.language:
            payload["language"] = self.language
        
        try:
            # Invoke SageMaker endpoint
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)  # Serialize payload to JSON string
            )

            
            # Parse response
            result = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            raise Exception(
                f"SageMaker endpoint error: {str(e)}\n"
                f"Make sure the endpoint '{self.endpoint_name}' exists and is in service in region '{self.region_name}'.\n"
                f"Also verify that your AWS credentials have permission to invoke SageMaker endpoints."
            )
        
        # Parse response - adapt to your endpoint's output format
        # Common formats include:
        # 1. Direct format: {"text": "...", "chunks": [...]}
        # 2. Nested format: {"predictions": {"text": "...", "chunks": [...]}}
        
        if 'predictions' in result:
            result = result['predictions']
        
        text = result.get('text', result.get('transcription', ''))
        
        # Handle different possible response formats for word-level timestamps
        word_locations = []
        chunks = result.get('chunks', result.get('words', []))
        
        for word_info in chunks:
            # Handle different timestamp formats
            if 'timestamp' in word_info:
                # Format: {"text": "word", "timestamp": [start, end]}
                timestamp = word_info['timestamp']
                start = timestamp[0] if timestamp[0] is not None else 0
                end = timestamp[1] if timestamp[1] is not None else start + 0.5
            elif 'start' in word_info and 'end' in word_info:
                # Format: {"word": "text", "start": 0.0, "end": 0.5}
                start = word_info['start']
                end = word_info['end']
            else:
                # No timestamps available
                start = 0
                end = 0
            
            word_text = word_info.get('text', word_info.get('word', '')).strip()
            
            word_locations.append({
                'word': word_text,
                'start_ts': start * self.sample_rate,
                'end_ts': end * self.sample_rate,
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
def get_api_asr_model(endpoint_name=None, region_name=None, language=None):
    """
    Factory function to create AWS SageMaker Whisper model.
    
    Usage:
        # Using default endpoint name (redparrot-whisper-base-provisioned)
        model = get_api_asr_model()
        
        # Or specify explicitly with language
        model = get_api_asr_model(endpoint_name='redparrot-whisper-base-provisioned', region_name='eu-central-1', language='en')
    """
    return WhisperAPIModel(endpoint_name=endpoint_name, region_name=region_name, language=language)

