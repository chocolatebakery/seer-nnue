# Atomic chess rules in this codebase (detailed)

- **Capture explosions**
  - A capture (including en passant) triggers an explosion centered on the **destination square (mv.to())** plus its 8 surrounding squares (king's neighbourhood/attack pattern).
  - For normal captures: destination square = captured piece square.
  - For en passant: destination square = where the capturing pawn moves to, NOT where the captured pawn is.
  - All non-pawn pieces in the blast radius are removed, for both sides.
  - Pawns are only removed if they are at the EXACT center of the explosion (destination square); pawns on adjacent squares survive the blast.
  - A capture is **illegal** if the blast would include our own king. Otherwise, it is legal even if the enemy king sits on the target (kings "die" by the blast, not the capture).

- **Checks and legality**
  - Direct check is evaluated like orthodox chess using normal attacks. A move is illegal if, after the move, our king is in direct check **unless** both kings are adjacent (see below).
  - If both kings end adjacent after the move, they do not give check to each other; direct-check legality is skipped in that case.
  - Indirect (explosion) exposure is allowed: you may move into an "indirect check" as long as our king is not in direct check in the resulting position.
  - **IMPORTANT**: Captures that would explode our own king are ILLEGAL, even if the enemy king also dies in the blast. You cannot sacrifice your own king.

- **Kings**
  - Move generation for kings produces only quiet king moves (no captures). Adjacency of kings is legal; they are not considered in check when touching.
  - Castling is allowed only if rights exist, path squares are empty, the king is not in direct check on the start/path/final squares, and the destination is also free of direct check. Blast rules still apply to the final position.

- **En passant**
  - En passant is generated/validated normally, but the explosion is centred on the **destination square** (where the capturing pawn moves to), NOT on the square of the captured pawn.
  - The captured pawn is removed from its original square (mv.enpassant_sq()).
  - The capturing pawn explodes at the destination square.
  - Pawns adjacent to the explosion center survive.
  - The move is illegal if that blast would hit our king.

- **Pawns**
  - Push/attack directions follow orthodox chess; promotions are to knight/bishop/rook/queen.
  - Pawns are removed ONLY if they are at the exact center of an explosion (the destination square of a capture).
  - Pawns on squares adjacent to the explosion center survive the blast.

- **Move validation summary**
  - From/to must follow normal move legality for the piece.
  - Captures require a target (or valid en passant). Capturing a king is only rejected if the resulting blast would hit our king.
  - Promotions only from last rank by pawns to N/B/R/Q.
  - A move is rejected if it leaves our king in direct check (unless kings end adjacent) or if it explodes our king.
