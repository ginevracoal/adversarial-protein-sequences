import os 
from Bio import Seq
import pickle as pkl
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment


def load_pfam(filepath, filename, max_model_tokens, max_tokens=None, align=True):

    path = os.path.join(filepath, filename)
    # min_seq_length = min([len(record.seq) for record in SeqIO.parse(path, format='fasta')])

    if max_tokens is None:
        max_tokens = max_model_tokens-2

    if align:

        data = []
        for record in SeqIO.parse(path, format='fasta'):

            if len(record.seq) < max_tokens:
                record.seq = str(record.seq).ljust(max_tokens, '.')
            else:
                record.seq = Seq.Seq(record.seq[:max_tokens])

            assert len(record.seq) == max_tokens

            data.append(record)

        aligned_data = MultipleSeqAlignment(data)

        data = []
        for record in aligned_data:
            name = record.name #.split('.')[0]
            sequence = str(record.seq)
            data.append((name, sequence))

        print(f"\nSequences aligned and cut to {max_tokens} tokens")

    else:

        file = open(path, 'r')
        file_contents = file.read().split('>')[0:-1]

        data = []
        avg_seq_lenght = 0
        for row in file_contents:
            if row!='':
                name, sequence = row.split('\n')[0:-1]
                # name = name.split('.')[0]
                sequence = sequence[:max_tokens]
                data.append((name, sequence))
                avg_seq_lenght += len(sequence)
                assert len(sequence)<=max_tokens

        avg_seq_lenght = int(avg_seq_lenght/len(data))

        print(f"\nSequences cut to {max_tokens} tokens")

    return data, max_tokens


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