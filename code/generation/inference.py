import torch
import argparse
import pandas as pd
import os
import sys
import numpy as np
from tqdm import tqdm

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def generate_preconditions(model, args):
    model_name = args.load_model.split("/")[-2]
    if args.test:
        with open("../peko_gen_test.txt", "r") as fin, \
                open(f"peko_generation_{model_name}.txt", "w") as fout:
            header = ["Reference", "Target Event", "Generated Precondition"]
            fout.write("\t".join(header) + "\n")
            for line in fin:
                row = line.strip().split("\t")

                print("Target Event: ", row[1])
                out = [row[0], row[1]]

                token_ids = model.tokenizer.encode(row[1] + " <sep>")
                sent = model.generate(
                        token_ids,
                        max_len=args.maxlen,
                        beam_size=args.beam_size)
                out.append(sent)
                fout.write("\t".join(out) + "\n")

    elif args.atomic:
        with open("../data/atomic_samples.txt", "r") as fin, \
                open(f"atomic_generation_{model_name}.txt", "w") as fout:
            fout.write("Seed\tGenerated Precondition\n")
            data = []
            for i, line in enumerate(fin):

                line = line.strip()

                data.append(line)
            idxs = np.random.permutation(len(data))
            for i, idx in enumerate(idxs[:1000]):
                line = data[idx]

                print("Seed Event: ", line)
                out = [line]

                token_ids = model.tokenizer.encode(line)
                sent = model.generate(
                        token_ids,
                        max_len=args.maxlen,
                        beam_size=args.beam_size)
                out.append(sent)
                fout.write("\t".join(out) + "\n")

    elif args.tellmewhy:
        splits = ['tellmewhy_val_set_peko_input.json', 'tellmewhy_test_set_peko_input.json', 'tellmewhy_train_set_peko_input.json']
        out_fnames = ['val.json', 'test.json', 'train.json']
        folder_path = '../../data/'
        for out_idx, split in enumerate(splits):
            df = pd.read_json(os.path.join(folder_path, split))
            print(f'Split: {split}')
            preconditions = []
            for idx, row in tqdm(df.iterrows()):
                model_input = row['model_input']
                token_ids = model.tokenizer.encode(model_input + " <sep>")
                sent = model.generate(
                        token_ids,
                        max_len=args.maxlen,
                        beam_size=args.beam_size)
                preconditions.append(sent)
            df['peko_generation'] = preconditions
            df.to_json('/home/yklal95/tellmewhy-ebl/data/tellmewhy_peko/'+out_fnames[out_idx], orient="records")

    else:
        with open("val_simple.txt", "r") as fin:
            for line in fin:
                text = line.strip()

                print("Event: ", text)
                token_ids = model.tokenizer.encode(text)
                sent = model.generate(
                        token_ids,
                        max_len=args.maxlen,
                        beam_size=args.beam_size)
                for i, s in enumerate(sent):
                    print(f"\t Rank {i+1}: {s}")

    return


def main(args):
    model_file = args.load_model
    model = torch.load(model_file)

    model.decoding_scheme = args.decoding
    model.eval()
    with torch.no_grad():
        generate_preconditions(model, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--load_model', type=str)
    parser.add_argument('-b', '--beam_size', type=int, default=10)
    parser.add_argument('-d', '--decoding', type=str, default='beam')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-tmw', '--tellmewhy', action='store_true')
    parser.add_argument('-a', '--atomic', action='store_true')
    parser.add_argument('-l', '--maxlen', type=int, default=60)

    args = parser.parse_args()

    main(args)
