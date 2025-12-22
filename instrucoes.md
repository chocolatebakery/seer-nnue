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

## rescore

Rescore reescreve um .bin com novos valores, mantendo as features.

Exemplo:

```bash
seer datagen rescore --in data.bin --out data_rescore.bin --mode search --nodes 200000 --threads 8
```

Opcoes principais:
- `--mode search|tb|tb_or_search` (default: search)
- `--nodes N` (default: 200000)
- `--depth N` (0 = disable)
- `--threads N`
- `--progress N` (0 = disable)
- `--tb-path PATH`
- `--tb-pieces 6`

## filtros e variedade

Presets:
- `minimal`: reis presentes, posicao legal, nao terminal, sem mate em 1.
- `balanced`: minimal + dedup + controlo leve de check/capture.
- `quiet`: semelhante ao original (sem check, static_eval == q_eval) + dedup.

Opcionais:
- `--require-capture-prob X` (0..1) exige contacto (check/capture) com probabilidade X.
- `--min-pieces N` evita finais cedo.
- `--dedup N` ou `--dedup-hash-mb M` para dedup simples.
- `--eval-limit N` (default 6144) define o corte de score para terminar o jogo.
- `--fixed-depth N` (default 6) limita a depth por jogada.
- `--fixed-nodes N` (default 5120) limita os nodes por jogada.
- `--allow-mate-in-one` aceita posicoes com mate em 1.
- `--no-quiet-filter` desliga o filtro quiet do preset `quiet`.

## conversao bin -> txt

Use `datagen-example-usage/bin_to_txt.py` para inspecionar o `.bin`:

```bash
python datagen-example-usage/bin_to_txt.py --bin data.bin --out data.txt
```
