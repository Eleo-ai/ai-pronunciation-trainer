"""
Test script to test the Whisper endpoint
"""
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.serializers import JSONSerializer
import time
import base64


endpoint_name = 'whisper-endpoint-gpu'
predictor = HuggingFacePredictor(endpoint_name=endpoint_name)
predictor.serializer = JSONSerializer()
predictor.content_type = 'application/json'

with open("sample.flac", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

print("\n" + "="*60)
print("Test 1: Auto-detect language (language=None)")
print("="*60)
req_start = time.time()
payload = {
    "audio": audio_base64
    # No language specified - will auto-detect
}
result = predictor.predict(payload)
req_end = time.time()
duration = req_end - req_start
print(f"Time taken: {duration:.2f} seconds")
print(f"Result: {result}")

print("\n" + "="*60)
print("Test 2: Specify English language")
print("="*60)
req_start = time.time()
payload = {
    "audio": audio_base64,
    "language": "en"
}
result = predictor.predict(payload)
req_end = time.time()
duration = req_end - req_start
print(f"Time taken: {duration:.2f} seconds")
print(f"Result: {result}")

print("\n" + "="*60)
print("Test 3: Specify Spanish language")
print("="*60)
req_start = time.time()
payload = {
    "audio": audio_base64,
    "language": "es"
}
result = predictor.predict(payload)
req_end = time.time()
duration = req_end - req_start
print(f"Time taken: {duration:.2f} seconds")
print(f"Result: {result}")

print("\n" + "="*60)
print("Test 4: With timestamps")
print("="*60)
req_start = time.time()
payload = {
    "audio": audio_base64,
    "timestamps": True
}
result = predictor.predict(payload)
print(result['segments'][0]['words'])
req_end = time.time()
duration = req_end - req_start
print(f"Time taken: {duration:.2f} seconds")
transcription = result.get('text', result.get('transcription', ''))
print(f"Result: {transcription}")
if 'words' in result:
    print(f"\nWord-level timestamps (showing first 10 words):")
    for word in result['words'][:10]:
        print(f"  [{word['start']:.2f}s - {word['end']:.2f}s]: '{word['word']}'")
    if len(result['words']) > 10:
        print(f"  ... and {len(result['words']) - 10} more words")
if 'segments' in result:
    print(f"\nTimestamp segments (showing first 3):")
    for seg in result['segments'][:3]:
        print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
    if len(result['segments']) > 3:
        print(f"  ... and {len(result['segments']) - 3} more segments")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
