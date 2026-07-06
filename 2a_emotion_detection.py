# emotion_detection.py
import requests


EMOTION_PREDICT_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
EMOTION_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def run_emotion_detection(text_to_analyze: str):
    """
    Envoie text_to_analyze à Watson NLP (EmotionPredict) et retourne la réponse JSON.
    """
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(EMOTION_PREDICT_URL, headers=EMOTION_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

from emotion_detection import run_emotion_detection

print(run_emotion_detection("I love this new technology"))
    