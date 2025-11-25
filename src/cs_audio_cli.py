import argparse
from src.compress import encode_audio_global
from src.reconstruct import decode_and_reconstruct

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    # Encode (compress)
    enc = sub.add_parser('encode')
    enc.add_argument('--wav', required=True)
    enc.add_argument('--out', required=True)
    enc.add_argument('--R', type=float, default=0.15)
    enc.add_argument('--seed', type=int, default=42)
    enc.add_argument('--frame_size', type=int, default=2048)
    enc.add_argument('--overlap', type=float, default=0.5)

    # Decode (reconstruct)
    dec = sub.add_parser('decode')
    dec.add_argument('--cs', required=True)
    dec.add_argument('--out', required=True)
    dec.add_argument('--solver', choices=['fista','lasso','omp'], default='fista')

    args = parser.parse_args()

    if args.cmd == 'encode':
        encode_audio_global(
            args.wav, args.out, R=args.R,
            seed=args.seed, frame_size=args.frame_size,
            overlap=args.overlap
        )
    elif args.cmd == 'decode':
        decode_and_reconstruct(
            args.cs, args.out,
            solver=args.solver
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
