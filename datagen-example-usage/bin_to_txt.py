#!/usr/bin/env python3
"""
Convert Seer atomic piece-list .bin to TXT (FEN|score|r).

Binary record format (little-endian):
  [u8 n][u8 stm][n * (u8 piece_code, u8 sq)][i16 score][i8 result]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import struct

import chess
import chess.variant


CODE_TO_SYMBOL = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
}
RESULT_CHAR = {0: "l", 1: "d", 2: "w"}

HDR = struct.Struct("<BB")
TAIL = struct.Struct("<hb")


def read_record(fh):
    hdr = fh.read(HDR.size)
    if not hdr:
        return None
    if len(hdr) != HDR.size:
        raise EOFError("Truncated record header")

    n, stm = HDR.unpack(hdr)
    if n < 2 or n > 32:
        raise ValueError(f"Invalid piece count n={n} (expected 2..32)")

    piece_bytes = fh.read(2 * n)
    if len(piece_bytes) != 2 * n:
        raise EOFError("Truncated piece list")

    tail = fh.read(TAIL.size)
    if len(tail) != TAIL.size:
        raise EOFError("Truncated tail")

    score, res = TAIL.unpack(tail)

    invalid = False
    seen_squares = 0
    white_kings = 0
    black_kings = 0

    for i in range(0, len(piece_bytes), 2):
        code = piece_bytes[i]
        sq = piece_bytes[i + 1]
        if code not in CODE_TO_SYMBOL or sq > 63:
            invalid = True
            break
        bit = 1 << sq
        if seen_squares & bit:
            invalid = True
            break
        seen_squares |= bit
        if code == 5:
            white_kings += 1
        elif code == 11:
            black_kings += 1

    if res not in RESULT_CHAR:
        invalid = True

    has_kings = (white_kings == 1 and black_kings == 1)

    return n, stm, piece_bytes, score, res, invalid, has_kings


def pieces_to_board(stm: int, piece_bytes: bytes) -> chess.Board:
    board = chess.variant.AtomicBoard(None)
    board.clear_board()
    board.turn = (stm == 1)
    board.castling_rights = 0
    board.ep_square = None
    for i in range(0, len(piece_bytes), 2):
        code = piece_bytes[i]
        sq = piece_bytes[i + 1]
        board.set_piece_at(sq, chess.Piece.from_symbol(CODE_TO_SYMBOL[code]))
    return board


def main():
    ap = argparse.ArgumentParser(description="Convert piece-list BIN to TXT (FEN|score|r).")
    ap.add_argument("--bin", required=True, help="Input .bin file.")
    ap.add_argument("--out", required=True, help="Output .txt file.")
    ap.add_argument("--limit", type=int, default=0, help="Max records to write (0 = all).")
    ap.add_argument("--strict", action="store_true", help="Stop on invalid records.")
    ap.add_argument(
        "--allow-no-kings",
        action="store_true",
        help="Allow records without both kings (default: skip).",
    )
    ap.add_argument(
        "--progress-updates",
        type=int,
        default=0,
        help="How many progress updates (0 disables).",
    )
    args = ap.parse_args()

    in_path = Path(args.bin)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.out)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_total = in_path.stat().st_size
    step_bytes = 0
    if args.progress_updates and args.progress_updates > 0:
        step_bytes = max(1, bytes_total // args.progress_updates)
    next_report = step_bytes

    total = 0
    written = 0
    skipped_invalid = 0
    skipped_no_kings = 0

    with in_path.open("rb") as fin, out_path.open("w", encoding="utf-8") as fout:
        while True:
            rec = read_record(fin)
            if rec is None:
                break
            total += 1
            n, stm, piece_bytes, score, res, invalid, has_kings = rec

            if invalid:
                skipped_invalid += 1
                if args.strict:
                    raise ValueError(f"Invalid record at #{total} in {in_path}")
                continue

            if not args.allow_no_kings and not has_kings:
                skipped_no_kings += 1
                continue

            board = pieces_to_board(stm, piece_bytes)
            res_char = RESULT_CHAR[res]
            fout.write(f"{board.fen()}|{score}|{res_char}\n")
            written += 1

            if args.limit and written >= args.limit:
                break

            if step_bytes and fin.tell() >= next_report:
                pct = min(100.0, 100.0 * fin.tell() / max(1, bytes_total))
                print(
                    f"{in_path.name}: {pct:6.2f}% | total={total:,} written={written:,} "
                    f"invalid={skipped_invalid:,} sem_reis={skipped_no_kings:,}"
                )
                next_report += step_bytes

    print(
        f"Done. total={total:,} written={written:,} invalid={skipped_invalid:,} sem_reis={skipped_no_kings:,}"
    )


if __name__ == "__main__":
    main()
