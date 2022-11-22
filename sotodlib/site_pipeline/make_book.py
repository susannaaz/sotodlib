import os, os.path as op
import argparse
from ..io.bookbinder import Bookbinder


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smurf', dest='smurffiles', nargs='+', type=str, required=True,
                        help='full path to SMuRF files')
    parser.add_argument('--hk', dest='hkfiles', nargs='+', type=str,
                        help='full path to HK files')
    parser.add_argument('--out-root', dest='out_root', default=".", type=str,
                        help='Path to output directory')
    parser.add_argument('--prefix', default="obs", help='prefix for output Book')
    parser.add_argument('--timestamp', dest='timestamp', type=int,
                        help='timestamp of the start of the observation')
    parser.add_argument('--tel-tube', dest='teltube', type=str,
                        help='ID of telescope and optics tube')
    parser.add_argument('--slot-flags', dest='slotflags', type=str,
                        help='1s and 0s indicating eligible stream_ids included in Book')
    parser.add_argument('--start-time', dest='start_time', type=int,
                        help='Timestamp of first sample (inclusive)')
    parser.add_argument('--end-time', dest='end_time', type=int,
                        help='Timestamp of last sample (inclusive)')
    parser.add_argument('--max-nchannels', dest='max_nchannels', type=int,
                        help='Largest number of channels in Book; used to determine file splits')
    if args is None:
        args = parser.parse_args(args)

    book_id = args.prefix
    if args.timestamp is not None:
        book_id += '_' + str(args.timestamp)
    if args.teltube is not None:
        book_id += '_' + args.teltube
    if args.slotflags is not None:
        book_id += '_' + args.slotflags
    if not op.exists(book_id):
        os.makedirs(book_id)

    B = Bookbinder(smurf_files = args.smurffiles,
                   hk_files = args.hkfiles,
                   out_root = args.out_root,
                   book_id = book_id,
                   session_id = args.timestamp,
                   stream_id = args.slotflags,
                   start_time = args.start_time,
                   end_time = args.end_time)
    B()


if __name__ == '__main__':
    print(main())
