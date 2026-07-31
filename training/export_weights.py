#!/usr/bin/env python3

"""

export_weights.py — dump a trained BaseConditionedGRUCell's weights from a

PyTorch checkpoint into raw binary files (row-major float32), for loading

directly into the C++ Eigen port.



For each weight tensor, writes <name>.bin containing just the raw float32

values in row-major order (Eigen's default MatrixXf storage is column-major,

so the C++ loader will need to either transpose on load or read with that

in mind -- see notes printed at the end of this script).



Also writes weights_meta.txt with shapes and vocab info needed to

reconstruct matrices/vectors on the C++ side.



Usage:

    python3 export_weights.py weights_v3_base_epoch14.pt export_h256/

"""



import argparse

import os



import torch





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint")

    parser.add_argument("outdir")

    args = parser.parse_args()



    os.makedirs(args.outdir, exist_ok=True)



    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    state = ckpt['model']

    qual_vocab = ckpt['qual_vocab']

    hidden_size = ckpt['hidden_size']



    # Only the cell.* weights are needed for the GRU recurrence itself.

    # out_proj.* would be needed too if/when you port the final

    # quality-score prediction layer -- exported here as well for

    # completeness, since you'll want it eventually for real inference.

    cell_keys = {k: v for k, v in state.items() if k.startswith('cell.')}

    out_proj_keys = {k: v for k, v in state.items() if k.startswith('out_proj.')}



    meta_lines = []

    meta_lines.append(f"qual_vocab_size {len(qual_vocab)}")

    meta_lines.append(f"hidden_size {hidden_size}")

    meta_lines.append(f"qual_vocab {''.join(qual_vocab)}")



    def dump_tensor(name, tensor):

        # tensor.shape is (rows, cols) for 2D, (n,) for 1D (bias vectors)

        arr = tensor.detach().numpy().astype('float32')

        flat_name = name.replace('.', '_')

        out_path = os.path.join(args.outdir, f"{flat_name}.bin")

        arr.tofile(out_path)  # raw floats, row-major (numpy default, C order)



        shape_str = 'x'.join(str(d) for d in arr.shape)

        meta_lines.append(f"tensor {flat_name} {shape_str} {out_path}")

        print(f"Wrote {out_path}  shape={arr.shape}")



    for name, tensor in {**cell_keys, **out_proj_keys}.items():

        dump_tensor(name, tensor)



    meta_path = os.path.join(args.outdir, "weights_meta.txt")

    with open(meta_path, 'w') as f:

        f.write('\n'.join(meta_lines) + '\n')

    print(f"\nWrote metadata to {meta_path}")



    print("\n--- IMPORTANT NOTE FOR C++ LOADING ---")

    print("These .bin files are raw float32 in ROW-MAJOR order (numpy/C default).")

    print("Eigen::MatrixXf is COLUMN-MAJOR by default. When loading in C++, either:")

    print("  (a) declare your matrices as Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>, or")

    print("  (b) load into a temporary row-major map, then assign into your normal MatrixXf")

    print("      (Eigen will handle the transpose-on-copy automatically).")

    print("Option (b) is simpler if you already declared everything as plain MatrixXf.")





if __name__ == "__main__":

    main()
