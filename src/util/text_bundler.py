import os


def bundle_text_label_pairs(data_dir_path):
    """
    :param data_dir_path: folder path where tagged data is stored
    :param label_type: Label that is going to be extracted along with sentence. ie: line_type and line_label
    :return: returns list of tupled sentence and line_type
    """
    # if label_type is None:
    #     label_type = 'line_type'

    result=[]

    for f in os.listdir(data_dir_path):
        data_file_path = os.path.join(data_dir_path, f)
        if os.path.isfile(data_file_path) and f.lower().endswith('.txt'):
            with open(data_file_path, mode='rt', encoding='utf-8') as file:
                for line in file:
                    line_type, line_label, sentence = line.strip().split('\t')
                    # if label_type == 'line_type':
                    #     result.append((sentence, line_type))
                    # else:
                    #     result.append((sentence, line_label))

                    # instead of processing line_type or line_label at one time, line_type and line_label processing is done
                    # at same time.
                    result.append((line_type, line_label, sentence))

    return result
