import azure.cognitiveservices.speech as speechsdk
import requests, uuid, json

#load from .env file
AUDIO_TO_TEXT_KEY = os.getenv("AUDIO_TO_TEXT_KEY")
TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
SPEECH_KEY = os.getenv("SPEECH_KEY")

def audio_to_text(audio_file_name):
    """
    Returns the recognized text from the audio file.
    """

    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(
        subscription=AUDIO_TO_TEXT_KEY, region="eastus"
    )
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_name)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print(
            "No speech could be recognized: {}".format(
                speech_recognition_result.no_match_details
            )
        )
        return speech_recognition_result.no_match_details
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
        return cancellation_details.error_details


def translator(input_text):
    """
    Translates the input text from English to Spanish.
    """
    # Add your key and endpoint
    endpoint = "https://api.cognitive.microsofttranslator.com"

    # location, also known as region.
    # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
    location = "westus2"

    path = "/translate"
    constructed_url = endpoint + path

    params = {
        "api-version": "3.0",
        "from": "en",
        "to": ["es"],
        "includeAlignment": True,
    }

    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        # location required if you're using a multi-service or regional (not global) resource.
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    # You can pass more than one object in body.
    body = [{"text": input_text}]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    print(
        json.dumps(
            response,
            sort_keys=True,
            ensure_ascii=False,
            indent=4,
            separators=(",", ": "),
        )
    )
    return response


def synthesize_from_text(text):
    """
    Synthesizes the text passed in as an argument and generates speech using Spanish voice.
    """


    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY, region="eastus"
    )
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name = "es-MX-HildaNeural"

    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if (
        speech_synthesis_result.reason
        == speechsdk.ResultReason.SynthesizingAudioCompleted
    ):
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
    return text
