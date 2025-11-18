import ModelInterfaces
import epitran
import eng_to_ipa


def get_phonem_converter(language: str):
    if language == 'de':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('deu-Latn'))
    elif language == 'en':
        phonem_converter = EngPhonemConverter()
    elif language == 'zh':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('ltc-Latn-bax'))
    elif language == 'cs':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('ces-Latn'))
    elif language == 'da':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('dan-Latn'))
    elif language == 'nl':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('nld-Latn'))
    elif language == 'fi':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('fin-Latn'))
    elif language == 'fr':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('fra-Latn'))
    elif language == 'de':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('deu-Latn'))
    elif language == 'it':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('ita-Latn'))
    elif language == 'es':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('spa-Latn'))
    elif language == 'ja':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('jpn-Hira'))
    elif language == 'ko':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('kor-Hang'))
    elif language == 'pl':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('pol-Latn'))
    elif language == 'pt':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('por-Latn'))
    elif language == 'no':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('nno-Latn'))
    elif language == 'pt':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('por-Latn'))
    elif language == 'sv':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('swe-Latn'))
    elif language == 'tr':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('tur-Latn'))
    else:
        raise ValueError('Language not implemented')

    return phonem_converter

class EpitranPhonemConverter(ModelInterfaces.ITextToPhonemModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, epitran_model) -> None:
        super().__init__()
        self.epitran_model = epitran_model

    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = self.epitran_model.transliterate(sentence)
        return phonem_representation


class EngPhonemConverter(ModelInterfaces.ITextToPhonemModel):

    def __init__(self,) -> None:
        super().__init__()

    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
