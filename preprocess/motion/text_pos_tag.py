"""Add HumanML3D-style part-of-speech tags to the narration files.

Each ``texts/<id>.txt`` becomes ``<sentence>#<word/POS ...>#0.0#0.0``: white space in the sentence is normalized and
nouns and verbs are lemmatized with spaCy ``en_core_web_sm`` (the same processing as the text-motion evaluator).

Usage:
    python -m preprocess.motion.text_pos_tag --text_dir datasets/ego3dlm/texts
"""
import argparse
import os
import re

import spacy


def process_text(nlp, sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list, pos_list = [], []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', required=True, help='directory of <id>.txt narration files (edited in place)')
    parser.add_argument('--out_dir', default=None, help='write tagged files here instead of in place')
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')
    out_dir = args.out_dir or args.text_dir
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(args.text_dir)):
        with open(os.path.join(args.text_dir, name)) as f:
            sentence = f.readline().split('#')[0]
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        words, pos = process_text(nlp, sentence)
        tokens = ' '.join(f'{w}/{p}' for w, p in zip(words, pos))
        with open(os.path.join(out_dir, name), 'w') as f:
            f.write(f'{sentence}#{tokens}#0.0#0.0')


if __name__ == '__main__':
    main()
