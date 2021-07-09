import os
import numpy as np
import torch

from argparse import ArgumentParser

def parser():

    ap = ArgumentParser()
    ap.add_argument('--embeddings', required=True, type=str)
    ap.add_argument('--embedding_names', required=True, type=str)
    ap.add_argument('--np', action='store_true')
    ap.add_argument('--output_dir' , required=True, type=str)
    ap.add_argument('--prefix' , required=False, default=None, type=str)

    return ap.parse_args()

def save_tsv_np(embedding_npy, metadata_npy):
    out_embedding = os.path.splitext(os.path.basename(embedding_npy))[0] + '.tsv'
    out_metadata = os.path.splitext(os.path.basename(metadata_npy))[0] + '.tsv'
    file_embedding = open(out_embedding, 'w')
    file_metadata = open(out_metadata, 'w')

    words = np.load(metadata_npy)
    embeddings = np.load(embedding_npy)

    for word, embedding in zip(words, embeddings):
        file_embedding.write('\t'.join([str(e) for e in embedding]) + '\n')
        file_metadata.write(word + '\n')

    file_embedding.close()
    file_metadata.close()

def save_tsv_pt(embedding_pt, output_dir, prefix):
    '''
    assumes there is one embedding in this file
    '''

    tensor = torch.load(embedding_pt)
    os.makedirs(output_dir, exist_ok=True)

    label = tensor['label']

    if prefix is not None:
        label = prefix + label

    out_embedding = os.path.splitext(os.path.basename(embedding_pt))[0] + '.tsv'
    if os.path.isfile(out_embedding):
        return
    out_metadata = os.path.splitext(os.path.basename(embedding_pt))[0] + '-metadata'+ '.tsv'
    out_embedding = os.path.join(output_dir, out_embedding)
    out_metadata = os.path.join(output_dir, out_metadata)

    embedding = tensor['mean_representations'][33]
    print(embedding.shape)

    with open(out_embedding, 'w') as dst:
        dst.write('\t'.join([str(e.item()) for e in embedding]) + '\n')

    with open(out_metadata, 'w') as dst:
        dst.write(label + '\n')



if __name__ == '__main__':

    args = parser()
    if args.np:
        save_tsv_np(args.embeddings, args.embedding_names)
    else:
        save_tsv_pt(args.embeddings, args.output_dir, args.prefix)
