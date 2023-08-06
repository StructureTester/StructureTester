import warnings
from collections import deque

import numpy as np
from nltk.tree import ParentedTree
from stanfordcorenlp import StanfordCoreNLP

from utils.central_word_util import get_central_word

warnings.filterwarnings('ignore')
print('Connecting to NLP server...')
nlp = StanfordCoreNLP('http://localhost', port=9000)
print('Connected to NLP server.')


def level_order_traversal(tree):
    queue = deque([(tree, 0)])
    traversal = []
    while queue:
        node, level = queue.popleft()
        if len(traversal) <= level:
            traversal.append([])
        traversal[level].append(node)
        for child in node:
            if isinstance(child, ParentedTree):
                queue.append((child, level + 1))
    return traversal


def get_node_with_label(tree, label):
    for subtree in tree.subtrees():
        if len(subtree.leaves()) == 1 and subtree.leaves()[0] == label:
            return subtree


def get_node_path(node):
    path = []
    while node is not None:
        path.append(node.label())
        node = node.parent()
    path.reverse()
    return path


def get_diff_vector(tree1, tree2):
    vector = np.zeros(20)

    level_order1 = level_order_traversal(tree1)
    level_order2 = level_order_traversal(tree2)

    i = 0
    while i < len(level_order1) and i < len(level_order2):
        level1 = level_order1[i]
        level2 = level_order2[i]
        count_diff = 0
        for i2 in range(min(len(level1), len(level2))):
            node1 = level1[i2]
            node2 = level2[i2]
            if node1.parent() is None or node2.parent() is None:
                continue
            if node1.label() == node2.label() and node1.parent().label() == node2.parent().label():
                continue
            else:
                count_diff += 1
        count_diff += abs(len(level1) - len(level2))
        if i >= 20:
            break
        vector[i] = count_diff
        i += 1

    return vector


def get_diff_vector_with_keyword(tree_path1, tree_path2):
    vector_with_keyword = np.zeros(20)
    for i in range(1, min(20, min(len(tree_path1), len(tree_path2)))):
        if tree_path1[i] != tree_path2[i] or tree_path1[i - 1] != tree_path2[i - 1]:
            vector_with_keyword[i] = 1
    for i in range(min(20, min(len(tree_path1), len(tree_path2))), max(len(tree_path1), len(tree_path2))):
        vector_with_keyword[i] = 1
    return vector_with_keyword


def get_constituency_tree(conSentence):
    parse_tree = nlp.parse(conSentence)
    nlp.close()
    return parse_tree


def build_tree(edges):
    nodes = {}
    for label, start, end in edges:
        if start not in nodes:
            nodes[start] = {'label': label, 'children': []}
        if end not in nodes:
            nodes[end] = {'label': None, 'children': []}
        nodes[start]['children'].append(end)
        nodes[end]['label'] = label
    root = None
    for node in nodes.values():
        if node['label'] == 'ROOT':
            root = node
            break
    return tree_to_string(root, nodes)


def tree_to_string(node, nodes):
    if not node['children']:
        return node['label']
    else:
        children_str = ' '.join(tree_to_string(nodes[child], nodes) for child in node['children'])
        return f"({node['label']} {children_str})"


def get_dependency_tree(depSentence):
    depSentence = depSentence.replace('.', ',')
    if depSentence[-1] == ',':
        depSentence = depSentence[0:-2] + '.'
    dep_tree = nlp.dependency_parse(depSentence)
    tree = build_tree(dep_tree)
    nlp.close()
    return tree


def get_tree_distance(parse1, parse2, s1, s2, i):
    parse_tree1 = ParentedTree.fromstring(parse1)
    parse_tree2 = ParentedTree.fromstring(parse2)

    diff_vector = get_diff_vector(parse_tree1, parse_tree2)

    key_word1 = get_central_word(s1, i)
    key_word2 = get_central_word(s2, i)
    parent_node1 = get_node_with_label(parse_tree1, key_word1)
    parent_node2 = get_node_with_label(parse_tree2, key_word2)

    path1 = get_node_path(parent_node1)
    path2 = get_node_path(parent_node2)

    diff_vector2 = get_diff_vector_with_keyword(path1, path2)

    return np.concatenate([diff_vector, diff_vector2])
