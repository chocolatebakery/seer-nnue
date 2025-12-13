# MonicaBang - Regras de Atomic Chess

## Tabuleiro e notacao
- Tabuleiro 8x8 e posicao inicial identicos ao xadrez classico; protocolo UCI.
- FEN padrao de 6 campos; halfmove/fullmove e casa de en-passant sao lidos/escritos.
- Reis podem iniciar ou ficar adjacentes; FEN com reis encostados e aceito.

## Objetivo e termino
- Vitoria ao remover o rei adversario por explosao de captura.
- Captura direta do rei ou xeque-mate (peças nao precisam de proteçao porque o rei nao captura) sao o fim de jogo;
- Empate por afogamento (sem lances legais), repeticao tripla ou regra dos 50 lances (100 meia-jogadas).
- Nao ha deteccao extra de material insuficiente alem desses criterios.

## Explosoes (capturas)
- Explosao so ocorre em capturas (incluindo promocoes e en passant).
- Centro da explosao e a casa da peca capturada; em en passant e a casa do peao que avancou duas casas.
- Area afetada: casa capturada + 8 casas adjacentes (mascara de rei).
- Remocoes:
  - A peca capturadora sempre explode (mesmo peoes e em en passant).
  - Todas as pecas nao-peoes na area de explosao sao removidas (torres, cavalos, bispos, damas e reis).
  - Peoes so saem se estiverem na casa capturada; peoes em casas adjacentes sobrevivem.
- Casas vazias nao importam; a casa de origem ja esta vazia e nao explode.
- Se a explosao eliminar o proprio rei o lance e ilegal; se eliminar o rei adversario, o lance e valido e encerra a partida.
- Direitos de roque sao perdidos se uma torre na casa original for destruida pela explosao.

## Legalidade e "xeque" atomico
- Um lance so e legal se, apos executado, o adversario nao tiver captura legal que remova o seu rei (checado por simulacao).
- Estar "em xeque"(cheque indireto) = o adversario ter uma captura legal que explode o seu rei; para sair basta eliminar essa captura possivel.
- Reis podem ficar encostados; ataques de rei nao contam como ameaca na geracao, ou seja reis encostados nunca ficarao em xeque.
- Reis nao conseguem capturar: qualquer captura com rei faz o proprio rei explodir, logo esses lances sao rejeitados.
- Capturar o rei adversario e permitido (desde que o seu rei sobreviva a explosao).

## Movimentos especiais
- Roque: requer torre na casa de origem e casas de passagem livres. Nao pode rocar sob captura imediata do adversario e cada passo do rei e testado contra explosoes possiveis; se algum passo deixaria o rei explodivel o roque e ilegal. Reis podem atravessar casas adjacentes ao rei inimigo.
- En passant: alvo definido apos avanco duplo. A explosao usa a casa do peao capturado; a peca capturadora tambem explode na casa de destino. Peoes adjacentes nao explodem. Ilegal se o rei de quem captura explodir; vitoria se o rei adversario explodir.
- Promocoes: apenas para dama, torre, bispo ou cavalo; promocoes de captura usam a mesma logica de explosao.
- Avanco duplo de peao e atualizacao do halfmove seguem o xadrez padrao (reseta em capturas ou movimento de peao).

## Resultados e empates
- Rei proprio removido -> lance invalido; rei adversario removido -> vitoria imediata.
- Sem lances legais e rei vivo -> empate (afogamento).
- Regra dos 50 lances dispara em 100 meia-jogadas; repeticao tripla detectada por chave de posicao.
