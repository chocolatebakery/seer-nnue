# instrucoes

Este repositorio inclui o datagen atomic embutido no binario `seer`.

## uso rapido

```bash
seer datagen --out data.bin --format bin --samples 2000000 --seed 1 --threads 8 --max-moves 256
```

Defaults principais:
- `--filter balanced`
- `--plies-min 8` / `--plies-max 16`
- `--require-capture-prob 0.2` (so em balanced)
- `--dedup 1000000` (so em balanced/quiet)
- `--threads 1`
- `--progress 2000` (0 = disable)

## seeds

- Startpos: use `--startpos` (default quando nao ha seeds).
- EPD/FEN: use `--epd seeds.epd` (pode repetir). Linhas apos `;` sao ignoradas. EPD e lido em streaming (nao carrega tudo em RAM); ao chegar ao fim, volta ao inicio.

Exemplo:

```bash
seer datagen --out out.bin --format bin --samples 500000 --epd seeds.epd --threads 4
```

## filtros e variedade

Presets:
- `minimal`: reis presentes, posicao legal, nao terminal, sem mate em 1.
- `balanced`: minimal + dedup + controlo leve de check/capture.
- `quiet`: semelhante ao original (sem check, static_eval == q_eval) + dedup.

Opcionais:
- `--require-capture-prob X` (0..1) exige contacto (check/capture) com probabilidade X.
- `--min-pieces N` evita finais cedo.
- `--dedup N` ou `--dedup-hash-mb M` para dedup simples.

## conversao bin -> txt

Use `datagen-example-usage/bin_to_txt.py` para inspecionar o `.bin`:

```bash
python datagen-example-usage/bin_to_txt.py --bin data.bin --out data.txt
```
