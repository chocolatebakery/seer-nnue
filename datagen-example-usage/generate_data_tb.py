#!/usr/bin/env python3
"""
Gera posições Atomic aleatórias com número fixo de peças (inclui sempre 2 reis).
Opcionalmente usa TB Atomic para rotular WDL; caso contrário escreve draw/score=0.

Formato de saída:
  --format bin: piece-list binário [u8 n][u8 stm][n*(u8 code,u8 sq)][i16 score][i8 result]
  --format txt: linhas "FEN|score|result_char"
  --format epd: linhas "FEN;" (apenas posições, útil para cutechess)

Validação (por defeito):
  - Descarta posições inválidas (board.is_valid()).
  - Descarta posições com rei em check (para evitar FENs "impossíveis").
    Use --allow-check para aceitar checks.

Regras aplicadas:
- Não permite peões duplos por cor (mesma coluna).
- Reis podem estar adjacentes.
- Sem roques/EP.
"""

import argparse
import os
import random
import struct
from typing import Optional

import chess
import chess.variant


PIECE_CODE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}
RESULT_BIN = {"l": 0, "d": 1, "w": 2}

HDR = struct.Struct("<BB")   # n, stm
TAIL = struct.Struct("<hb")  # score (i16), result (i8)


def random_position(num_pieces: int, rng: random.Random) -> Optional[chess.Board]:
    """Cria AtomicBoard com num_pieces total (inclui 2 reis); rejeita peões duplos por cor."""
    if num_pieces < 2 or num_pieces > 32:
        raise ValueError("num_pieces deve estar entre 2 e 32")

    b = chess.variant.AtomicBoard(None)
    b.clear_board()
    b.castling_rights = 0
    b.ep_square = None

    remaining = num_pieces - 2
    pool = ["P", "N", "B", "R", "Q"] * 4 + ["p", "n", "b", "r", "q"] * 4
    pieces = ["K", "k"] + rng.sample(pool, remaining)
    squares = rng.sample(range(64), len(pieces))

    pawns_white_files = set()
    pawns_black_files = set()

    for sym, sq in zip(pieces, squares):
        file = sq % 8
        if sym == "P":
            if file in pawns_white_files:
                return None
            pawns_white_files.add(file)
        if sym == "p":
            if file in pawns_black_files:
                return None
            pawns_black_files.add(file)
        b.set_piece_at(sq, chess.Piece.from_symbol(sym))

    b.turn = bool(rng.randint(0, 1))
    return b


def is_acceptable_position(board: chess.Board, allow_check: bool, allow_double_check: bool) -> bool:
    if not board.is_valid():
        return False
    turn = board.turn
    board.turn = chess.WHITE
    white_in_check = board.is_check()
    board.turn = chess.BLACK
    black_in_check = board.is_check()
    board.turn = turn
    if not allow_check:
        return (not white_in_check) and (not black_in_check)
    if not allow_double_check and white_in_check and black_in_check:
        return False
    return True


def wdl_from_tb(tb, board: chess.Board) -> Optional[str]:
    """Se TB disponível, devolve 'w','d','l'; caso contrário None."""
    try:
        w = tb.probe_wdl(board)
    except Exception:
        return None
    if w >= 1:
        return "w"
    if w <= -1:
        return "l"
    return "d"


def write_record_bin(fh, board: chess.Board, score: int, res_char: str) -> None:
    pm = board.piece_map()
    n = len(pm)
    stm = 1 if board.turn == chess.WHITE else 0
    fh.write(HDR.pack(n, stm))
    items = [(PIECE_CODE[pc.symbol()], sq) for sq, pc in pm.items()]
    items.sort()
    for code, sq in items:
        fh.write(struct.pack("<BB", code, sq))
    fh.write(TAIL.pack(score, RESULT_BIN[res_char]))


def write_record_txt(fh, board: chess.Board, score: int, res_char: str) -> None:
    fh.write(f"{board.fen()}|{score}|{res_char}\n")


def write_record_epd(fh, board: chess.Board) -> None:
    fh.write(f"{board.fen()};\n")


def main():
    ap = argparse.ArgumentParser(description="Gera posições Atomic aleatórias com N peças.")
    ap.add_argument("--pieces", type=int, required=True, help="Total de peças (inclui os 2 reis).")
    ap.add_argument("--samples", type=int, required=True, help="Quantos registos gerar.")
    ap.add_argument("--out", required=True, help="Ficheiro de saída (.bin ou .txt).")
    ap.add_argument("--format", choices=["bin", "txt", "epd"], default="bin", help="Formato de saída.")
    ap.add_argument("--seed", type=int, default=1, help="Semente RNG.")
    ap.add_argument("--tb", default=None, help="(Opcional) pasta das TBs Atomic para rotular WDL.")
    ap.add_argument("--logit-scale", type=int, default=1024, help="Escala para score quando usa TB.")
    ap.add_argument("--allow-check", action="store_true", help="Permitir posições com rei em check.")
    ap.add_argument("--allow-double-check", action="store_true", help="Permitir ambos os reis em check.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tb = None
    if args.tb:
        import chess.syzygy
        tb = chess.syzygy.open_tablebase(args.tb, VariantBoard=chess.variant.AtomicBoard)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open_mode = "wb" if args.format == "bin" else "w"

    with open(args.out, open_mode) as fh:
        written = 0
        step = max(1, args.samples // 100)
        while written < args.samples:
            board = random_position(args.pieces, rng)
            if board is None:
                continue
            if not is_acceptable_position(board, args.allow_check, args.allow_double_check):
                continue

            res_char = "d"
            score = 0
            if tb is not None:
                res = wdl_from_tb(tb, board)
                if res is None:
                    continue
                res_char = res
                score = args.logit_scale if res_char == "w" else (-args.logit_scale if res_char == "l" else 0)

            if args.format == "bin":
                write_record_bin(fh, board, score, res_char)
            elif args.format == "txt":
                write_record_txt(fh, board, score, res_char)
            else:  # epd
                write_record_epd(fh, board)

            written += 1
            if written % step == 0:
                pct = 100.0 * written / args.samples
                print(f"{written}/{args.samples} ({pct:.1f}%)")

    print(f"OK: {args.samples} exemplos gravados em {args.out}")


if __name__ == "__main__":
    main()
