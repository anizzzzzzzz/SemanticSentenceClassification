from bs4 import BeautifulSoup
from tika import parser
import re

def parsing_resume(temp_filepath):
    """
    :param temp_filepath: path of file that is temporarily created in temp folder of project
    :return: This method will convert content in html format, returns only body content of html ignoring all unnecessary data in string
    """
    file_data = parser.from_file(temp_filepath, xmlContent=True)
    if file_data['status'] == 200:
        isParsed = True
        soup = BeautifulSoup(file_data['content'], features="html.parser")
        content = cleaning_text(soup.body.get_text())
        return content, isParsed
    else:
        isParsed=False
        return '',isParsed


def cleaning_text(content):
    """
    :param content: parsed resume content which will be cleaned based on line break and tab
    :return: will return clean content
    """
    content = re.sub('[\\n]{2,}', '\n', content)
    content = re.sub('[\\t]{1,}', ' ', content)
    return content