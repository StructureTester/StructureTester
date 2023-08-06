import spacy
from transformers import BertTokenizer, BertForMaskedLM, pipeline

nlp = spacy.load("zh_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")


def find_mask_word(sentence):
    doc = nlp(sentence)
    tagged_words = [(token.text, token.pos_) for token in doc]
    noun = None
    for word, tag in tagged_words:
        if tag.startswith('N'):
            noun = word
            break

    return noun


def generate_similar_sentences(n, input_str):
    mask_word = find_mask_word(input_str)

    if mask_word is None or input_str == len(mask_word):
        return []

    mask_sentence = input_str.replace(mask_word, tokenizer.mask_token, 1)

    fill_mask = pipeline(
        "fill-mask",
        top_k=n,
        model=model,
        tokenizer=tokenizer,
    )
    choices = fill_mask(mask_sentence)
    return [choice['sequence'].replace(" ", "") for choice in choices]
