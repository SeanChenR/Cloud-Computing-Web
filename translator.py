#  參考文件
#  https://learn.microsoft.com/en-us/python/api/overview/azure/ai-translation-text-readme?view=azure-python-preview
import configparser
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem

config = configparser.ConfigParser()
config.read('config.ini')

REGION = config['TranslateAzure']['region'] # 區域
KEY = config['TranslateAzure']['key'] # 金鑰
ENDPOINT = config['TranslateAzure']['endpoint'] # 服務端點

def translator_azure(src, dst, targets):

    text_translator = TextTranslationClient(
        endpoint=ENDPOINT,
        credential=TranslatorCredential(KEY, REGION)
    )

    targets = [InputTextItem(text=targets)]

    response = text_translator.translate(content=targets, to=[dst], from_parameter=src)

    return response[0]['translations'][0]['text']

if __name__ == '__main__':
    src = 'zh'
    dst = 'zh-hant'
    targets = '草地上的几只狗'
    print(translator_azure(src, dst, targets))