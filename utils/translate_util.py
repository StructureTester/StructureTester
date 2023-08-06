import requests
import yaml
from google.cloud import translate_v2 as translate


def google_translate(text, target):
    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)
    return result


def deepl_translate(text, target):
    with open('../config/translate.yaml', 'r') as f:
        config = yaml.safe_load(f)
    auth_key = config['deepl']['auth_key']
    f.close()

    result = requests.get(
        'https://api.deepl.com/v2/translate',
        params={
            'auth_key': auth_key,
            'target_lang': target,
            'text': text,
        },
    )
    translated_text = result.json()['translations'][0]['text']
    return translated_text


def get_translate_text(text, target, translator_type):
    if translator_type == 'google':
        return google_translate(text, target)
    elif translator_type == 'deepl':
        return deepl_translate(text, target)
    else:
        raise Exception('No such type of translation')
