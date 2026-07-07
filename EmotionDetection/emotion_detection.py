"""
Emotion detection module using Watson NLP.
"""

import requests


def emotion_detector(text_to_analyze):
    """
    Analyze the emotion of the provided text using Watson NLP.

    Args:
        text_to_analyze (str): Text to analyze.

    Returns:
        dict: Formatted emotion scores and dominant emotion.
    """
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    result = response.json()

    emotion_scores = result["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
        "dominant_emotion": dominant_emotion
    }
