import requests
import json

EMOTION_PREDICT_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
EMOTION_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def run_emotion_detection(text_to_analyze: str):
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(EMOTION_PREDICT_URL, headers=EMOTION_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()


def emotion_detector(text_to_analyze: str):
    # 1) Obtenir la réponse (déjà un dict si tu utilises response.json())
    result = run_emotion_detection(text_to_analyze)

    # 2) Récupérer les scores
    # Structure observée dans ton output:
    # result['emotionPredictions'][0]['emotion'] -> {anger, disgust, fear, joy, sadness}
    emotion_scores = result["emotionPredictions"][0]["emotion"]

    anger_score = emotion_scores["anger"]
    disgust_score = emotion_scores["disgust"]
    fear_score = emotion_scores["fear"]
    joy_score = emotion_scores["joy"]
    sadness_score = emotion_scores["sadness"]

    # 3) Dominant emotion = max score
    score_map = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score
    }
    dominant_emotion = max(score_map, key=score_map.get)

    # 4) Format de sortie demandé
    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion
    }