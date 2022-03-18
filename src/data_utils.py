import os 


def filter_pfam(filepath, filename):

    path = os.path.join(filepath, filename)
    file = open(path, 'r')

    file_contents = file.read().split('>')[0:-1]

    data = []
    for row in file_contents:
        if row!='':
            name, sequence = row.split('\n')[0:-1]
            name = name.split('.1')[0]
            data.append((name, sequence))

    return data