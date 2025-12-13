# Atomic rules implemented in Seer

- **Explosions**: Any capture (including en passant) triggers a blast on the captured square plus its 8 neighbours (`king` moves). The blast removes all non-pawn pieces in that mask for both sides, including kings; pawns are immune. If the blast would hit our king, the move is illegal.
- **Checks**: Direct check works like standard chess, except if both kings end up adjacent after the move; adjacent kings are mutually “not in check”. Indirect checks (exposing the king to a future blast) are allowed. Moves leaving us in direct check are illegal unless both kings die in the blast.
- **Capturing the king**: A direct capture of the enemy king is allowed only if the blast does not include our king (king dies via the blast, not the capture itself).
- **King moves**: The king never captures in move generation; only quiet king moves are generated. Adjacent kings are legal; they do not check each other.
- **Castling**: Standard occupancy and rights are required, plus no direct check on the starting square or along the king’s path. The destination must also be free of direct check; blast rules still apply.
- **En passant**: Treated as a capture with the blast centered on the en-passant square; illegal if our king is in that blast.
- **Pawns**: Push/attack directions follow orthodox chess. Promotions are orthodox (to N/B/R/Q).
- **Explosion immunity**: Pawns survive blasts; all other pieces (including kings) are removed if in the blast.
