#  參考文件
#  https://learn.microsoft.com/en-us/python/api/overview/azure/cognitiveservices-vision-computervision-readme?view=azure-python-previous
import configparser
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from translator import translator_azure

config = configparser.ConfigParser()
config.read('config.ini')

KEY = config['VisionAzure']['key']  # 金鑰
ENDPOINT = config['VisionAzure']['endpoint']  # 服務端點

def recognizer_azure(image, lang, max_describe):

    client = ComputerVisionClient(
        endpoint=ENDPOINT,
        credentials=CognitiveServicesCredentials(KEY)
    )
    if lang == "zh-hant":
        vision_lang = "zh"
    else:
        vision_lang = lang

    analysis = client.describe_image(url=image, max_descriptions=max_describe, language=vision_lang)

    if lang == 'zh-hant':
        zh_hant_describe = translator_azure("zh", "zh-hant", analysis.captions[0].text)
        return zh_hant_describe, analysis.captions[0].confidence
        # print(f"{zh_hant_describe} with score {analysis.captions[0].confidence}")
    else:
        return analysis.captions[0].text, analysis.captions[0].confidence
        # print(f"{analysis.captions[0].text} with score {analysis.captions[0].confidence}")

if __name__ == '__main__':
    recognize_result, recognize_score = recognizer_azure("https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg", "es", 3)
    print(recognize_result, round(recognize_score*100, 2))