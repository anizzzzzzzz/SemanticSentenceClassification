from __root__ import getPath
from src.SentenceClassifier import SentenceClassifier
from src.util.tika_parser import parsing_resume


def main():
    root_dir = getPath()

    resume = input('Enter the path of resume : ')
    content, isParsed = parsing_resume(resume)

    if isParsed is True:
        # print(content)

        parser = SentenceClassifier()
        parser.load_model(root_dir + '/models')
        prediction = parser.predict_resume_context(content)

        predicted_content = display_result(prediction)
        # print(parser.display_result())
        print(predicted_content)


def display_result(all):
    text = ''
    for index, type, label, edu in all['education']:
        text += 'Education : {}\t{}\t{}\t{}\n'.format(index, type, label, edu)
    text += '------------------------------------------------\n'
    for index, type, label, exp in all['experience']:
        text += 'Experience : {}\t{}\t{}\t{}\n'.format(index, type, label, exp)
    text += '------------------------------------------------\n'
    for index, type, label, proj in all['project']:
        text += 'Project : {}\t{}\t{}\t{}\n'.format(index, type, label, proj)
    text += '------------------------------------------------\n'
    for index, type, label, skill in all['skill']:
        text += 'Skills : {}\t{}\t{}\t{}\n'.format(index, type, label, skill)
    text += '------------------------------------------------\n'
    for index, type, label, personal in all['skill']:
        text += 'Personal : {}\t{}\t{}\t{}\n'.format(index, type, label, personal)
    text += '------------------------------------------------\n'

    return text


if __name__ == '__main__':
    main()