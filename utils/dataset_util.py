import xlrd
import xlwt
from tqdm import tqdm

from utils.bert_util import generate_similar_sentences
from utils.con_dep_tree_util import get_constituency_tree, get_dependency_tree, get_tree_distance
from utils.translate_util import get_translate_text


def get_dataset_for_label(path, translator_type):
    file = open(path, 'r')
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('data')

    lines = file.readlines()
    raw_data = ''
    for line in lines:
        raw_data += line

    data_list = raw_data.split('。')
    excel_index = 0
    for index in tqdm(range(1, len(data_list))):
        data = data_list[index - 1]
        data = data + "。"
        sheet.write(excel_index, 1, data)
        translate_data = get_translate_text(data, 'en', translator_type)
        sheet.write(excel_index, 2, translate_data)
        excel_index += 1

        similar_sentences = generate_similar_sentences(10, data)
        for sentence in similar_sentences:
            sheet.write(excel_index, 1, sentence)
            translate_data = get_translate_text(sentence, 'en', translator_type)
            sheet.write(excel_index, 2, translate_data)
            excel_index += 1
        excel_index += 1

    excel_path = path.replace('.txt', '.xlsx')
    workbook.save(excel_path)
    file.close()


def get_dataset(path, tree_type, i):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name('data')
    vectors = []
    flags = []
    sentences = []
    originals = []
    original_temp = []
    original = ''
    original_tree = None
    for index in tqdm(range(sheet.nrows)):
        if original == '':
            original = sheet.cell_value(index, 2)
            original_temp = [sheet.cell_value(index, 1), original]
            if tree_type == 'con':
                original_tree = get_constituency_tree(original)
            else:
                original_tree = get_dependency_tree(original)
            continue
        if len(sheet.cell_value(index, 1)) == 0:
            original = ''
            continue
        sentence = sheet.cell_value(index, 2)
        original_sentence = sheet.cell_value(index, 1)
        flag = sheet.cell_value(index, 0)
        sentences.append([original_sentence, sentence])
        originals.append(original_temp)
        if tree_type == 'con':
            sentence_tree = get_constituency_tree(sentence)
            diff_vector = get_tree_distance(original_tree, sentence_tree, original, sentence, i)
            flags.append(flag)
            vectors.append(diff_vector)
        else:
            sentence_tree = get_dependency_tree(sentence)
            diff_vector = get_tree_distance(original_tree, sentence_tree, original, sentence, i)
            flags.append(flag)
            vectors.append(diff_vector)

    return [vectors, flags, sentences, originals]
