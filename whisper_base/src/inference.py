import whisper
import json
import os
import io
import base64
import tempfile
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_audio_bytes(audio_bytes, sr=16000):
    """
    Load audio from bytes using multiple fallback methods.
    Returns numpy array at 16kHz sample rate.
    """
    import scipy.signal

    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=True) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            audio, sample_rate = sf.read(temp_audio.name)
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Resample to 16kHz if needed
            if sample_rate != sr:
                num_samples = int(len(audio) * sr / sample_rate)
                audio = scipy.signal.resample(audio, num_samples)
            np_array = audio.astype(np.float32)
            print(
                f"Loaded audio with soundfile, shape: {np_array.shape}, sample_rate: {sample_rate}")
            return np_array
    except Exception as e:
        print(f"soundfile failed: {e}")

    raise RuntimeError(
        "All audio loading methods failed. Please ensure audio is in a supported format (WAV, FLAC, OGG) or that ffmpeg is available for other formats like MP3.")


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model = whisper.load_model(os.path.join(model_dir, 'base.pt'))
    model = model.to(DEVICE)
    print(f'whisper model has been loaded to this device: {model.device.type}')
    return {'model': model}


def input_fn(request_body, request_content_type):
    """
    Takes in request and transforms it to necessary input type.
    Accepts either:
    1. Raw numpy array (for backward compatibility)
    2. JSON with 'audio' (base64 encoded audio file or list) and optional 'language'
    """
    try:
        print(f"input_fn called with content_type: {request_content_type}")
        if request_content_type == 'application/json':
            request_data = json.loads(request_body)

            # Extract audio data
            if isinstance(request_data, dict):
                audio_data = request_data.get('audio')
                # None for auto-detection
                language = request_data.get('language', None)
                # Include timestamps in response (default: False)
                timestamps = request_data.get('timestamps', False)

                # Handle different audio input formats
                if isinstance(audio_data, list):
                    # Direct numpy array as list
                    np_array = np.array(audio_data, dtype=np.float32)
                    print(f"Loaded audio from list, shape: {np_array.shape}")
                else:
                    # Base64 encoded data
                    audio_bytes = base64.b64decode(audio_data)
                    print(
                        f"Decoded base64 audio, size: {len(audio_bytes)} bytes")
                    np_array = load_audio_bytes(audio_bytes)

                data_input = torch.from_numpy(np_array)
                print(
                    f"Converted to tensor, shape: {data_input.shape}, dtype: {data_input.dtype}")
                return {'audio': data_input, 'language': language, 'timestamps': timestamps}
            else:
                raise ValueError(
                    "JSON input must be a dictionary with 'audio' key")
        else:
            # raw numpy array
            np_array = np.load(io.BytesIO(request_body))
            np_array = np_array.astype(np.float32)
            data_input = torch.from_numpy(np_array)
            return {'audio': data_input, 'language': None, 'timestamps': False}
    except Exception as e:
        print(f"ERROR in input_fn: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_fn(input_data, model_dict):
    """
    SageMaker model server invokes `predict_fn` on the return value of `input_fn`.

    Return predictions
    """
    try:
        print(f"predict_fn called")
        audio_tensor = input_data['audio']
        language = input_data['language']
        include_timestamps = input_data['timestamps']
        print(
            f"Audio tensor shape: {audio_tensor.shape}, language: {language}, timestamps: {include_timestamps}")

        # Ensure float32 dtype for whisper processing
        audio = whisper.pad_or_trim(audio_tensor.flatten()).to(DEVICE).float()
        print(
            f"Padded/trimmed audio shape: {audio.shape}, device: {audio.device}")

        if include_timestamps:
            # Use transcribe for word-level timestamps
            print("Using transcribe mode with timestamps")
            result = model_dict['model'].transcribe(
                audio.cpu().numpy(),
                language=language,
                word_timestamps=True,
                fp16=False
            )
            return {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language']
            }
        else:
            # Use decode for faster transcription without timestamps
            print("Using decode mode without timestamps")
            mel = whisper.log_mel_spectrogram(audio)
            print(f"Mel spectrogram shape: {mel.shape}")
            options = whisper.DecodingOptions(
                language=language, without_timestamps=True, fp16=False)
            output = model_dict['model'].decode(mel, options)
            print(
                f"Decode completed, text: {output.text[:50] if output.text else 'empty'}...")
            return {'text': str(output.text)}
    except Exception as e:
        print(f"ERROR in predict_fn: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    try:
        print(f"output_fn called with content_type: {content_type}")
        if isinstance(predictions, dict):
            if 'segments' in predictions:
                all_words = [word for segment in predictions['segments']
                             for word in segment.get('words', [])]
                result = json.dumps({
                    "transcription": predictions['text'],
                    "language": predictions['language'],
                    "segments": predictions['segments'],
                    "words": all_words
                })
            else:
                result = json.dumps({"transcription": predictions['text']})
        else:
            result = json.dumps({"transcription": predictions})
        print(f"output_fn returning: {result[:100]}...")
        return result
    except Exception as e:
        print(f"ERROR in output_fn: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
