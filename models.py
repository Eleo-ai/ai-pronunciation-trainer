from ModelInterfaces import IASRModel


def getASRModel(language: str, use_whisper: bool = True, use_api: bool = True) -> IASRModel:
    """
    Get ASR model for speech recognition using OpenAI Whisper API.

    Args:
        language: Language code ('de', 'en', 'fr') - used for reference only
        use_whisper: Always True (kept for backward compatibility)
        use_api: Always True (kept for backward compatibility)

    Returns:
        WhisperAPIModel instance
    """
    # Always use OpenAI Whisper API
    from whisper_api_wrapper import WhisperAPIModel
    return WhisperAPIModel()


def getTTSModel(language: str):
    """
    TTS functionality removed - use external TTS API if needed.
    Kept for backward compatibility but raises NotImplementedError.
    """
    raise NotImplementedError(
        "Local TTS models are no longer supported. "
        "Please use an external TTS API like AWS Polly, Google TTS, or ElevenLabs."
    )


def getTranslationModel(language: str):
    """
    Translation functionality removed - use external translation API if needed.
    Kept for backward compatibility but raises NotImplementedError.
    """
    raise NotImplementedError(
        "Local translation models are no longer supported. "
        "Please use an external translation API like Google Translate or DeepL."
    )
