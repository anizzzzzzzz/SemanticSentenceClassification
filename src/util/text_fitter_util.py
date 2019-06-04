import collections
import os

from src.util.text_tokenizer import word_tokenizer


def fit_text(data_dir_path, max_vocab_size = None):
    # if label_type is None:
    #     label_type = 'line_label'
    if max_vocab_size is None:
        max_vocab_size = 10000

    counter = collections.Counter()
    max_len = 0
    line_types = dict()
    line_labels = dict()
    for f in os.listdir(data_dir_path):
        data_file_path = os.path.join(data_dir_path, f)
        if os.path.isfile(data_file_path) and f.lower().endswith('.txt'):
            file = open(data_file_path, mode='rt', encoding='utf-8')

            for line in file:
                line_type, line_label, sentence = line.strip().split('\t')
                tokens = [x.lower() for x in word_tokenizer(sentence)]
                for token in tokens:
                    counter[token] += 1
                max_len = max(max_len, len(tokens))
                # if label_type == 'line_type':
                #     label = line_type
                # else:
                #     label = line_label
                # if label not in labels:
                #     labels[label] = len(labels)

                # instead of processing line_type or line_label at one time, line_type and line_label processing is done
                # at same time.
                if line_type not in line_types:
                    line_types[line_type] = len(line_types)
                if line_label not in line_labels:
                    line_labels[line_label] = len(line_labels)
            file.close()

    word2idx = collections.defaultdict(int)
    for idx, word in enumerate(counter.most_common(max_vocab_size)):
        word2idx[word] = idx
    idx2word = {v : k for k,v in word2idx.items()}
    vocab_size = len(word2idx) + 1

    model = dict()
    model['word2idx'] = word2idx
    model['idx2word'] = idx2word
    model['vocab_size'] = vocab_size
    model['max_len'] = max_len
    model['line_types'] = line_types
    model['line_labels'] = line_labels

    return model
