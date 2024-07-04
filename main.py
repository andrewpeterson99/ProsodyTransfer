from azure_functions import translator

import numpy as np
import json
import speech_recognition as sr

r = sr.Recognizer()

import gtts
from pydub import AudioSegment
import stable_whisper
from difflib import SequenceMatcher
import soundfile as sf
import os

from pydub.utils import make_chunks
from functools import reduce

# constants/hyperparameters
VOLUME_MULTIPLER = 0.02
SENTENCE_AUDIO_DIR = os.path.join(os.getcwd(), "sentence_audio")
RESULTS_AUDIO_DIR = os.path.join(os.getcwd(), "results_audio")

NORMALIZED_DB = MIN_NORMALIZED_DB, MAX_NORMALIZED_DB = [-32.0, -22.0]
SAMPLE_RATE = 100

# set to True if you want to load files from sentence_audio and results_audio
# use False if you want to record audio from microphone
LOADING_FILES = False


def match_target_amplitude(sound, target_dBFS):
    """
    Refines a sound's amplitude to a target dBFS.

    sound: AudioSegment object
    target_dBFS: int
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def sound_slice_normalize(sound, sample_rate, target_dBFS):
    """
    Normalizes the volume of the sound.

    sound: AudioSegment object
    sample_rate: int
    target_dBFS: tuple of the form (min, max)
    """

    def max_min_volume(min, max):
        for chunk in make_chunks(sound, sample_rate):
            if chunk.dBFS < min:
                yield match_target_amplitude(chunk, min)
            elif chunk.dBFS > max:
                yield match_target_amplitude(chunk, max)
            else:
                yield chunk

    return reduce(lambda x, y: x + y, max_min_volume(target_dBFS[0], target_dBFS[1]))


def similar(a, b):
    """
    Returns a ratio of how similar a and b are.

    a: string
    b: string
    """
    return SequenceMatcher(None, a, b).ratio()


def get_average_dBFS(audio):
    """
    Returns the average dBFS of the audio.

    audio: AudioSegment object
    """
    return audio.rms


def transfer_volume(src_audio, tgt_audio):
    """
    Transfers volume from src_audio to tgt_audio.

    src_audio: AudioSegment object
    tgt_audio: AudioSegment object
    """
    src_avg_dBFS = get_average_dBFS(src_audio)

    dBFS_difference = src_avg_dBFS - (get_average_dBFS(tgt_audio))

    adjusted_audio = tgt_audio
    if dBFS_difference > 0:
        adjusted_audio = tgt_audio + (VOLUME_MULTIPLER * dBFS_difference)

    return adjusted_audio


def transfer_prosody(src_audio, tgt_audio):
    """
    Transfers prosody from src_audio to tgt_audio.

    src_audio: AudioSegment object
    tgt_audio: AudioSegment object
    """

    transferred = transfer_volume(src_audio, tgt_audio)
    # you can add more prosody transfer functions here, e.g. pitch transfer
    # pass in 'transferred' to the next function as the tgt_audio

    return transferred


def prepare_prosody(src_bundle, tgt_bundle, alignment):
    """
    Isolates the audio segments on a word-level basis and calls transfer_prosody() on each segment.
    Returns the stitched tgt_audio.

    src_bundle: object with properties: raw_text, audio, words
    tgt_bundle: object with properties: raw_text, audio, words
    alignment: string of the form "0:3-0:5 7:9-7:8 11:13-16:19 15:21-10:14"
    """

    # get the highest number in src alignments
    src_end = 0
    for element in alignment.split(" "):
        src_end = max(src_end, int(element.split("-")[0].split(":")[1]))

    # get the highest number in tgt alignments
    tgt_end = 0
    for element in alignment.split(" "):
        tgt_end = max(tgt_end, int(element.split("-")[1].split(":")[1]))

    # get length of src_audio and tgt_audio
    src_length = len(AudioSegment.from_file(src_bundle["audio"]))
    tgt_length = len(AudioSegment.from_file(tgt_bundle["audio"]))

    # set last timestamp to be slightly past whisper's last timestamp
    src_bundle["words"][-1]["end"] += src_length
    tgt_bundle["words"][-1]["end"] += tgt_length

    if src_bundle["words"][-1]["end"] > src_length:
        src_bundle["words"][-1]["end"] = src_length
    if tgt_bundle["words"][-1]["end"] > tgt_length:
        tgt_bundle["words"][-1]["end"] = tgt_length

    # we need to keep a list that contains the tgt_audio segments with their tgt_start time (where the audio segments are all sorted by tgt_start time)
    tgt_audio_segments = []

    # the elements in alignment look like the following: 0:3-0:5 (src_start:src_end-tgt_start:tgt_end)
    for element in alignment.split(" "):
        src_start = int(element.split("-")[0].split(":")[0])
        src_end = int(element.split("-")[0].split(":")[1])
        tgt_start = int(element.split("-")[1].split(":")[0])
        tgt_end = int(element.split("-")[1].split(":")[1])

        src_word = src_bundle["raw_text"][src_start : src_end + 1]
        tgt_word = tgt_bundle["raw_text"][tgt_start : tgt_end + 1]

        # get corr obj in words array using similar()
        src_word_obj = None
        tgt_word_obj = None
        for word in src_bundle["words"]:
            if similar(word["word"].strip(), src_word) > 0.8:
                src_word_obj = word
                break
        for word in tgt_bundle["words"]:
            if similar(word["word"].strip(), tgt_word) > 0.8:
                tgt_word_obj = word
                break

        if src_word_obj is None or tgt_word_obj is None:
            continue

        src_t1 = src_word_obj["start"]
        src_t2 = src_word_obj["end"]

        tgt_t1 = tgt_word_obj["start"]
        tgt_t2 = tgt_word_obj["end"]

        src_clip = AudioSegment.from_file(src_bundle["audio"])[src_t1:src_t2]
        tgt_clip = AudioSegment.from_file(tgt_bundle["audio"])[tgt_t1:tgt_t2]

        # do prosody transfer here
        tgt_clip = transfer_prosody(src_clip, tgt_clip)

        # append to tgt_audio_segments
        tgt_audio_segments.append((tgt_clip, tgt_t1))

    # sort tgt_audio_segments by tgt_start time to maintain tgt grammar
    tgt_audio_segments.sort(key=lambda x: x[1])

    # stitch tgt_audio_segments together into a new finished tgt_audio
    finished_tgt_audio = AudioSegment.empty()
    for segment in tgt_audio_segments:
        finished_tgt_audio += segment[0]

    return finished_tgt_audio


def parse_whisper_json(json_file):
    """
    Parses JSON from whisper.
    Returns a list of words, where each word is an object with properties: word, start, end
    word: the word
    start: the start time of the word in milliseconds
    end: the end time of the word in milliseconds
    """
    # read json file
    with open(json_file) as f:
        data = json.load(f)

    # get the words
    words = data["segments"][0]["words"]

    # make object for each word
    word_objects = []
    for word in words:
        word_objects.append(
            {
                "word": word["word"],
                "start": word["start"] * 1000,
                "end": word["end"] * 1000,
            }
        )

    return word_objects


def load_files():
    """
    Loads files for batch prosody transfer.
    """
    model = stable_whisper.load_model("base")
    for file in os.listdir(SENTENCE_AUDIO_DIR):
        if file.endswith(".wav") == False:
            continue
        print("Processing " + file)
        audio = AudioSegment.from_file(os.path.join(SENTENCE_AUDIO_DIR, file))
        audio = sr.AudioData(audio.raw_data, audio.frame_rate, audio.sample_width)

        src_text = r.recognize_whisper(audio).strip()
        print("Source text: " + src_text)

        translation_response = translator(src_text)
        tgt_text = translation_response[0]["translations"][0]["text"]
        print("Target text: " + tgt_text)
        alignment = translation_response[0]["translations"][0]["alignment"]["proj"]
        print("Alignment: " + alignment)

        tgt_audio = gtts.gTTS(tgt_text, lang="es")
        tgt_audio_name = file.split(".")[0] + "_tts.mp3"
        tgt_audio.save(os.path.join(RESULTS_AUDIO_DIR, tgt_audio_name))

        src_whisper_result = model.transcribe(os.path.join(SENTENCE_AUDIO_DIR, file))
        src_whisper_result.save_as_json(
            os.path.join(SENTENCE_AUDIO_DIR + "/json", file.split(".")[0] + ".json")
        )

        tgt_whisper_result = model.transcribe(
            os.path.join(RESULTS_AUDIO_DIR, tgt_audio_name)
        )
        tgt_whisper_result.save_as_json(
            os.path.join(
                RESULTS_AUDIO_DIR + "/json", tgt_audio_name.split(".")[0] + ".json"
            )
        )

        src_words = parse_whisper_json(
            os.path.join(SENTENCE_AUDIO_DIR + "/json", file.split(".")[0] + ".json")
        )
        tgt_words = parse_whisper_json(
            os.path.join(
                RESULTS_AUDIO_DIR + "/json", tgt_audio_name.split(".")[0] + ".json"
            )
        )

        src_bundle = {
            "raw_text": src_text,
            "audio": os.path.join(SENTENCE_AUDIO_DIR, file),
            "words": src_words,
        }

        tgt_bundle = {
            "raw_text": tgt_text,
            "audio": os.path.join(RESULTS_AUDIO_DIR, tgt_audio_name),
            "words": tgt_words,
        }

        transferred_tgt_audio = prepare_prosody(src_bundle, tgt_bundle, alignment)

        # write to file
        new_audio_signal = (
            np.array(transferred_tgt_audio.get_array_of_samples(), dtype=np.float32)
            / 32768.0
        )
        sf.write(
            os.path.join(RESULTS_AUDIO_DIR, file.split(".")[0] + "_transferred.wav"),
            new_audio_signal,
            transferred_tgt_audio.frame_rate,
        )


def use_microphone():
    """
    Records audio from microphone and performs prosody transfer.
    """
    with sr.Microphone() as source:
        print("Calibrating...")

        print("Okay, go!")
        audio = r.listen(source)
        # save the audio to a temporary file
        with open("microphone_results.wav", "wb") as f:
            f.write(audio.get_wav_data())
        print("Done recording")

        audio = AudioSegment.from_file("microphone_results.wav")
        audio = sr.AudioData(audio.raw_data, audio.frame_rate, audio.sample_width)
        src_text = r.recognize_whisper(audio).strip()

        # get translation
        translation_response = translator(src_text)
        tgt_text = translation_response[0]["translations"][0]["text"]
        alignment = translation_response[0]["translations"][0]["alignment"]["proj"]

        print("Alignment: " + alignment)

        tgt_audio = gtts.gTTS(tgt_text, lang="es")
        tgt_audio.save("mic_tgt_audio.mp3")

        model = stable_whisper.load_model("base")
        src_whisper_result = model.transcribe("microphone_results.wav")
        src_whisper_result.save_as_json("mic_src_audio.json")

        tgt_whisper_result = model.transcribe("mic_tgt_audio.mp3")
        tgt_whisper_result.save_as_json("mic_tgt_audio.json")

        src_words = parse_whisper_json("mic_src_audio.json")
        tgt_words = parse_whisper_json("mic_tgt_audio.json")

        src_bundle = {
            "raw_text": src_text,  # from recognize_whisper
            "audio": "microphone_results.wav",
            "words": src_words,  # whisper's timestamps/word info
        }
        tgt_bundle = {
            "raw_text": tgt_text,  # from translator
            "audio": "mic_tgt_audio.mp3",
            "words": tgt_words,  # whisper's timestamps/word info
        }

        transferred_tgt_audio = prepare_prosody(src_bundle, tgt_bundle, alignment)

        # # write to file
        new_audio_signal = (
            np.array(transferred_tgt_audio.get_array_of_samples(), dtype=np.float32)
            / 32768.0
        )  # scale to between [-1.0, 1.0]
        # play audio
        sf.write(
            "mic_transferred_temp.wav",
            new_audio_signal,
            transferred_tgt_audio.frame_rate,
        )
        os.system("mic_transferred_temp.wav")


def main():
    if LOADING_FILES == True:
        load_files()
        return
    else:
        use_microphone()
        return


if __name__ == "__main__":
    main()
