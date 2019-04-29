import json
import os

import requests
import prediction

api_url_base = os.getenv("API_BASE_SEARCH")
GOOGLE_SEARCH_APIKEY = os.getenv("GOOGLE_SEARCH_APIKEY")
GOOGLE_SEARCH_CONTEXT = os.getenv("GOOGLE_SEARCH_CONTEXT")

predictedClasses = {
    'sandals': 0,
    'sneakers': 1,
    'high_heels': 2,
    'boots': 3
}


def get_base_url():
    return '{0}'.format(api_url_base)


def suggest(filePath):
    predictions = prediction.predict(filePath)
    return {
        "i": predictions,
        "suggestions": getSuggestionsInfo(int(predictions['classes']))
    }


def getSuggestionsInfo(predictClass):

    api_url = get_base_url()

    params = {
        'key': GOOGLE_SEARCH_APIKEY,
        'cx': GOOGLE_SEARCH_CONTEXT,
        'q': buildQueryPerClass(predictClass)
    }

    response = requests.get(
        api_url,
        params=params)

    if response.status_code == 200:
        data = json.loads(response.content.decode('utf-8'))
        parseData = parseResponseSuggestions(data["items"])
        return parseData
    else:
        return None


def parseResponseSuggestions(data):
    sugg = []
    for x in data:
        item = {
            "title": x["title"],
            "link": x["link"],
            "displayLink": x["displayLink"]
        }
        sugg.append(item)
    return sugg


def buildQueryPerClass(pred):
    return {
        0: 'buy Sandals',
        1: 'buy Sneakers',
        2: 'buy High Heels',
        3: 'buy Boots'
    }[pred]
