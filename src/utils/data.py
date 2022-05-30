import os 
from Bio import Seq
import pickle as pkl
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment


def load_msa(filepath, filename, max_model_tokens, max_tokens=None, alignment_char="-"):

    path = os.path.join(filepath, filename)

    if max_tokens is None:
        max_tokens = max_model_tokens-2
    else:
        max_tokens = min(max_tokens, max_model_tokens)

    file = open(path, 'r')
    file_contents = file.read().split('>')[0:-1]

    data = []
    avg_seq_lenght = 0
    for row in file_contents:
        if row!='':
            name = row.split('\n')[0]
            sequence = row.split(name)[1].replace('\n', '')
            sequence = sequence[:max_tokens]

            data.append((name, sequence))
            avg_seq_lenght += len(sequence)
            assert len(sequence)<=max_tokens

    avg_seq_lenght = int(avg_seq_lenght/len(data))

    if len(sequence)<max_tokens:
        max_tokens = len(sequence)

    print(f"\nmax tokens = {max_tokens}")
    return data, max_tokens

def load_pfam(filepath, filename, max_model_tokens, max_tokens=None, align=False, alignment_char="-"):

    path = os.path.join(filepath, filename)

    if max_tokens is None:
        max_tokens = max_model_tokens-2
    else:
        max_tokens = min(max_tokens, max_model_tokens)

    if align:

        raise NotImplementedError("check this")

        # data = []
        # for record in SeqIO.parse(path, format='fasta'):

        #     if "-" in record:
        #         raise ValueError("Sequences are already aligned")

        #     if len(record.seq) < max_tokens:
        #         record.seq = str(record.seq).ljust(max_tokens, alignment_char)
        #     else:
        #         record.seq = Seq.Seq(record.seq[:max_tokens])

        #     assert len(record.seq) == max_tokens

        #     data.append(record)

        # aligned_data = MultipleSeqAlignment(data)

        # data = []
        # for record in aligned_data:
        #     name = record.name
        #     sequence = str(record.seq)
        #     data.append((name, sequence))

    else:

        file = open(path, 'r')
        file_contents = file.read().split('>')[0:-1]

        data = []
        avg_seq_lenght = 0
        for row in file_contents:
            if row!='':
                name, sequence = row.split('\n')[0:-1]
                sequence = sequence[:max_tokens]
                data.append((name, sequence))
                avg_seq_lenght += len(sequence)
                assert len(sequence)<=max_tokens

        avg_seq_lenght = int(avg_seq_lenght/len(data))

    print(f"\nmax tokens = {max_tokens}")
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