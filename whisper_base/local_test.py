"""
Local test script to test inference.py functions before deploying to SageMaker
"""
import whisper
import json
import base64

from src.inference import input_fn, predict_fn, output_fn


def test_inference_locally():
    print("="*60)
    print("LOCAL INFERENCE TEST")
    print("="*60)

    print("\n[1] Loading model...")
    try:
        model_dict = {'model': whisper.load_model('base')}
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("   Note: For local testing, this downloads the Whisper base model")
        return

    print("\n[2] Loading audio file...")
    try:
        with open("sample.flac", "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        print(f"   ✓ Audio loaded and encoded to base64 (no preprocessing needed!)")
    except Exception as e:
        print(f"   ✗ Error loading audio: {e}")
        return

    test_cases = [
        {"name": "Auto-detect language", "payload": {"audio": audio_base64}},
        {"name": "English", "payload": {"audio": audio_base64, "language": "en"}},
        {"name": "Spanish", "payload": {"audio": audio_base64, "language": "es"}},
        {"name": "With timestamps", "payload": {
            "audio": audio_base64, "language": "en", "timestamps": True}},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[3.{i}] Testing: {test_case['name']}")
        print("-" * 60)

        try:
            # input_fn
            request_body = json.dumps(test_case['payload'])
            request_content_type = 'application/json'
            input_data = input_fn(request_body, request_content_type)
            print(f"   ✓ input_fn executed successfully")
            print(f"     - Language: {input_data['language']}")
            print(f"     - Audio shape: {input_data['audio'].shape}")

            # predict_fn
            prediction = predict_fn(input_data, model_dict)
            print(f"   ✓ predict_fn executed successfully")

            # output_fn
            output = output_fn(prediction, 'application/json')
            print(f"   ✓ output_fn executed successfully")

            result = json.loads(output)
            print(f"\n   📝 TRANSCRIPTION: {result['transcription']}")

            if 'language' in result:
                print(f"   🌍 DETECTED LANGUAGE: {result['language']}")

            if 'segments' in result:
                print(f"   ⏱️  SEGMENTS WITH TIMESTAMPS:")
                for seg in result['segments'][:3]:
                    print(
                        f"      [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
                if len(result['segments']) > 3:
                    print(
                        f"      ... and {len(result['segments']) - 3} more segments")

            if 'words' in result:
                print(f"   ⏱️  WORDS WITH TIMESTAMPS:")
                for word in result['words'][:10]:
                    print(
                        f"      [{word['start']:.2f}s - {word['end']:.2f}s]: '{word['word']}'")
                if len(result['words']) > 10:
                    print(
                        f"      ... and {len(result['words']) - 10} more words")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("LOCAL TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    test_inference_locally()
