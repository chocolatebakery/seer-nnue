# Atomic chess rules in this codebase (detailed)

- **Capture explosions**
  - A capture (including en passant) triggers an explosion on the captured square plus its 8 surrounding squares (king’s neighbourhood).
  - All non-pawn pieces in the blast are removed, for both sides; pawns are immune.
  - A capture is **illegal** if the blast would include our own king. Otherwise, it is legal even if the enemy king sits on the target (kings “die” by the blast, not the capture).

- **Checks and legality**
  - Direct check is evaluated like orthodox chess using normal attacks. A move is illegal if, after the move, our king is in direct check **unless** both kings are adjacent (see below).
  - If both kings end adjacent after the move, they do not give check to each other; direct-check legality is skipped in that case.
  - Indirect (explosion) exposure is allowed: you may move into an “indirect check” as long as our king is not in direct check in the resulting position.
  - If both kings die in a blast, the move is allowed.

- **Kings**
  - Move generation for kings produces only quiet king moves (no captures). Adjacency of kings is legal; they are not considered in check when touching.
  - Castling is allowed only if rights exist, path squares are empty, the king is not in direct check on the start/path/final squares, and the destination is also free of direct check. Blast rules still apply to the final position.

- **En passant**
  - En passant is generated/validated normally, but the explosion is centred on the en-passant square (where the pawn lands). The move is illegal if that blast would hit our king.

- **Pawns**
  - Push/attack directions follow orthodox chess; promotions are to knight/bishop/rook/queen.
  - Pawns are immune to explosions (never removed by blasts).

- **Move validation summary**
  - From/to must follow normal move legality for the piece.
  - Captures require a target (or valid en passant). Capturing a king is only rejected if the resulting blast would hit our king.
  - Promotions only from last rank by pawns to N/B/R/Q.
  - A move is rejected if it leaves our king in direct check (unless kings end adjacent) or if it explodes our king.
