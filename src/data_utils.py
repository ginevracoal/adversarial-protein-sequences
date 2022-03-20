import os 
import pickle as pkl


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


############
# pickling #
############


def save_to_pickle(data, filepath, filename):

    full_path=os.path.join(filepath, filename+".pkl")
    print("\nSaving pickle: ", full_path)
    os.makedirs(filepath, exist_ok=True)
    with open(full_path, 'wb') as f:
        pkl.dump(data, f)

def load_from_pickle(filepath, filename):

    full_path=os.path.join(filepath, filename+".pkl")
    print("\nLoading from pickle: ", full_path)
    with open(full_path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    return data