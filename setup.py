import random

from constants import *


class Record:
    def __init__(self, has_increased):
        self.has_increased = has_increased
        self.headlines = []

    def display(self):
        print('Has increased: ' + str(self.has_increased))
        for i in range(0, len(self.headlines), 1):
            print(self.headlines[i])


def get_data():
    data = []
    with open(processed_file, 'r') as f:
        for line in f:
            line = line.strip('-')
            aux_line = line.split(',')
            if len(aux_line) > 25 and aux_line[1].isdigit():
                new_record = Record(int(aux_line[1]))
                line = line.replace(',"b', '^')
                line = line.replace(',b', '^')
                line = line.split('^')
                if len(line) == 26:
                    for i in range(1, 26, 1):
                        words = line[i].strip('"').strip("'()")
                        new_record.headlines.append(words)
                    data.append(new_record)
    return data


def setup_dataset():
    data = get_data()
    index_of_record = 0
    for record in data:
        index_of_record = index_of_record + 1
        # establish whether to fit it onto the training set or the testing set
        path = 'data\\'
        coin_flip = random.randint(0, 5)
        if coin_flip == 1:
            path = path + 'test\\'
        else:
            path = path + 'train\\'
        if record.has_increased:
            path = path + 'increasing\\'
        else:
            path = path + 'decreasing\\'

        path = path + 'sample' + str(index_of_record) + '.txt'
        with open(path, 'w') as f:
            for headline in record.headlines:
                f.write(headline)


# some rows
def filter_bad_data():
    out_file = open(processed_file, 'w')
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines) - 1, 1):
            start_of_current_row = lines[i][:4]
            print(start_of_current_row)
            if start_of_current_row in accepted_line_beginnings:
                out_file.write(lines[i])
    out_file.close()


filter_bad_data()
setup_dataset()
