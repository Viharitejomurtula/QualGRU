#!/usr/bin/env python3

"""

extract_paired.py — build a paired base+quality dataset file from FASTQ or BAM.



Output format: one read per line, tab-separated: <bases>\t<qualities>



Usage:

    python3 extract_paired.py input.fastq output_paired.txt

    python3 extract_paired.py input.bam output_paired.txt

"""



import sys





def from_fastq(path, out_f):

    n = 0

    with open(path) as f:

        while True:

            header = f.readline()

            if not header:

                break

            seq = f.readline().strip()

            plus = f.readline()

            qual = f.readline().strip()

            if len(seq) != len(qual) or len(seq) < 2:

                continue

            out_f.write(f"{seq}\t{qual}\n")

            n += 1

    return n





def from_bam(path, out_f):

    import pysam

    n = 0

    bam = pysam.AlignmentFile(path, 'rb', check_sq=False)

    for read in bam:

        if read.query_sequence is None or read.query_qualities is None:

            continue

        seq = read.query_sequence

        qual = ''.join(chr(q + 33) for q in read.query_qualities)

        if len(seq) != len(qual) or len(seq) < 2:

            continue

        out_f.write(f"{seq}\t{qual}\n")

        n += 1

    return n





if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: extract_paired.py <input.fastq|input.bam> <output.txt>")

        sys.exit(1)



    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(out_path, 'w') as out_f:

        if in_path.endswith('.bam'):

            n = from_bam(in_path, out_f)

        else:

            n = from_fastq(in_path, out_f)

    print(f"Wrote {n} paired reads to {out_path}")
