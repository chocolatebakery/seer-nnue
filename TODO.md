# TODO - Plano de Otimizações Futuras

## Status Atual
- ✅ **Correção completa**: Sem illegal moves, atomic chess implementado corretamente
- ⚠️ **Performance**: NPS significativamente reduzido devido a resets constantes em capturas
- 🎯 **Foco atual**: Treinar NNUE com código correto (prioridade)

## Problema Principal de Performance

### Análise do Bottleneck
Atualmente, toda captura (incluindo en passant) executa operações pesadas:

**Localização**: `board.cc` linhas 1218-1246 (função `forward_()`)

```cpp
// STEP 5: Recompute 'all' bitboards from individual piece bitboards
copy.man_.white.all_ = copy.man_.white.pawn_ | ... | copy.man_.white.king_;
copy.man_.black.all_ = copy.man_.black.pawn_ | ... | copy.man_.black.king_;

// STEP 6: Recompute hashes from scratch
copy.man_.white.hash_ = 0;
copy.man_.white.pawn_hash_ = 0;
copy.man_.black.hash_ = 0;
copy.man_.black.pawn_hash_ = 0;

over_types([&](const piece_type pt) {
  for (const auto sq : copy.man_.white.get_plane(pt)) {
    copy.man_.white.hash_ ^= sided_manifest::w_manifest_src.get(pt, sq);
    if (pt == piece_type::pawn) {
      copy.man_.white.pawn_hash_ ^= sided_manifest::w_manifest_src.get(pt, sq);
    }
  }
});
// ... mesmo para black
```

**Impacto**:
- Atomic chess tem MUITAS capturas (é uma variante agressiva)
- Cada captura recalcula TODOS os hashes do zero
- Cada captura recalcula TODOS os bitboards `all_`
- Isso acontece em TODAS as posições exploradas durante a busca

### Estimativa de Custo
- **Xadrez normal**: ~5-10% das posições têm capturas
- **Atomic chess**: ~40-60% das posições têm capturas (muito mais agressivo)
- **Custo de reset completo**: 6 loops sobre todos os bitboards + iteração sobre todas as peças

---

## �� Otimizações Planejadas (Ordem de Prioridade)

### 1. **Incremental Hash Updates para Explosões** ⭐⭐⭐ (Maior impacto)

**Problema**: Recalculamos hash do zero para cada captura
**Solução**: Manter hash incremental mesmo durante explosões

**Abordagem**:
```cpp
// Em vez de:
copy.man_.white.hash_ = 0;
// ... recalcular tudo

// Fazer:
// 1. Remove peças explodidas do hash (XOR out)
for (const auto sq : blast & copy.man_.white.knight_) {
  copy.man_.white.hash_ ^= sided_manifest::w_manifest_src.get(piece_type::knight, sq);
}
// 2. Remove do bitboard
copy.man_.white.knight_ &= ~blast;
// ... para todas as peças
```

**Ganho estimado**: 30-50% de speedup em posições com capturas

**Complexidade**: MÉDIA
- Requer cuidado para manter consistência
- Precisa garantir que todas as peças explodidas são removidas do hash
- Testes extensivos necessários

**Arquivos afetados**:
- `src/chess/board.cc` (linhas 1169-1246)
- `src/chess/board_state.cc` (pode precisar de novos métodos helper)

---

### 2. **Bitboard `all_` Incremental** ⭐⭐ (Médio impacto)

**Problema**: Recalculamos `all_` bitboard do zero
**Solução**: Atualizar incrementalmente

**Abordagem**:
```cpp
// Em vez de:
copy.man_.white.all_ = copy.man_.white.pawn_ | copy.man_.white.knight_ | ...;

// Fazer:
// Já temos all_ do estado anterior, apenas remover peças explodidas
copy.man_.white.all_ &= ~blast;
copy.man_.black.all_ &= ~blast;
```

**Ganho estimado**: 5-10% de speedup

**Complexidade**: BAIXA
- Muito simples de implementar
- Baixo risco

**Arquivos afetados**:
- `src/chess/board.cc` (linhas 1218-1222)

---

### 3. **Cache de Explosion Masks** ⭐⭐ (Médio impacto)

**Problema**: Calculamos `explosion_mask(square)` repetidamente
**Solução**: Pre-computar todas as máscaras de explosão

**Abordagem**:
```cpp
// Criar tabela constexpr
constexpr std::array<square_set, 64> explosion_masks = [] {
  std::array<square_set, 64> masks{};
  for (int i = 0; i < 64; ++i) {
    masks[i] = king_attack_tbl.look_up(square{i}, square_set{});
  }
  return masks;
}();

// Uso:
const square_set blast = explosion_masks[mv.to().index()];
```

**Ganho estimado**: 10-15% de speedup em geração de moves e is_legal

**Complexidade**: BAIXA
- Implementação direta
- Sem risco de bugs

**Arquivos afetados**:
- `include/chess/tables.h` (adicionar nova tabela)
- `src/chess/board.cc` (usar tabela em vez de função)

---

### 4. **Otimizar NNUE Feature Updates** ⭐ (Baixo-médio impacto)

**Problema**: `feature_full_reset()` é chamado para TODAS as capturas
**Solução**: Implementar updates incrementais mesmo para capturas simples

**Abordagem**:
- Usar `feature_full_reset()` apenas quando explosion remove >2 peças
- Para capturas simples (1v1), usar incremental updates mesmo em atomic

**Complexidade**: ALTA
- Requer mudanças profundas na lógica NNUE
- Precisa rastrear quantas peças foram removidas
- Pode introduzir bugs sutis

**Ganho estimado**: 15-25% de speedup (mas apenas se bem implementado)

**Arquivos afetados**:
- `include/chess/board.h` (linhas 300-310)
- Toda a lógica de feature updates

**⚠️ Nota**: Deixar para DEPOIS do treino inicial da NNUE estar completo

---

### 5. **Parallel Piece Removal** ⭐ (Baixo impacto, apenas para multi-thread)

**Problema**: Removemos peças de forma sequencial
**Solução**: Usar SIMD ou operações paralelas

**Complexidade**: MUITO ALTA
**Ganho estimado**: 5-10% (apenas em contextos multi-thread)
**Prioridade**: BAIXA (otimização prematura)

---

### 6. **Ajustes de Search para Atomic Chess** ⭐⭐ (Médio impacto)

**Problema**: Parâmetros de search (futility pruning, etc.) estão tuneados para xadrez clássico
**Solução**: Ajustar parâmetros baseando-se em Fairy-Stockfish

**Fonte**: Fairy-Stockfish ajusta vários parâmetros para variantes explosivas:

#### 6.1 Futility Pruning Depth
```cpp
// Fairy-Stockfish (search.cpp)
if (!PvNode
    && depth < 9 - 3 * pos.blast_on_capture()  // Para atomic: depth < 6
    && eval - futility_margin(...) >= beta
    && eval < VALUE_KNOWN_WIN)
    return eval;
```

**Análise**:
- Fairy reduz depth threshold de **9 para 6** em atomic (33% redução)
- Seer atual: `futility_prune_depth()` retorna **5** (já conservador)
- **Conclusão**: Seer já é mais conservador que o Fairy atomic (5 vs 6)
- **Ação**: Manter depth=5 por enquanto, testar depth=6 se necessário

#### 6.2 Futility Move Count
```cpp
// Fairy-Stockfish
int futilityMoveCount = (3 + depth * depth) / (2 + pos.blast_on_capture());
// Para atomic: divide por 2 (metade dos moves considerados)
```

**Seer atual**: Não tem limitação de move count em futility pruning (usa margem fixa)

**Abordagem**:
- Implementar move count limit similar ao Fairy
- Testar se melhora playing strength

#### 6.3 Outras Possíveis Adaptações
- **Null Move Pruning**: Fairy pode ajustar depth reduction para atomic
- **Late Move Reductions**: Podem precisar de ajustes para capturas explosivas
- **Extensions**: Capturas que removem várias peças podem merecer extension

**Complexidade**: MÉDIA
- Requer testing extensivo para validar melhoria
- Mudanças localizadas em `search_worker.cc`
- Baixo risco de bugs (apenas tuning)

**Ganho estimado**: 5-15% de playing strength (depende de tuning)

**Arquivos afetados**:
- `src/search/search_worker.cc` (linhas 434-437 e outras)
- `include/search/search_constants.h` (linhas 85+)

**⚠️ Nota**: Fazer DEPOIS de NNUE estar treinado
- Requer baseline estável para medir melhorias
- Interação com NNUE pode afetar resultados
- Tuning de search é processo iterativo (requer muitos testes)

**Prioridade**: MÉDIA-ALTA (mas apenas após Fase 1 completa)

---

## 📊 Roadmap de Implementação

### Fase 1: Treino NNUE (ATUAL) 🎯
- ✅ Código correto sem illegal moves
- 🔄 Treinar NNUE com TB pura
- 🔄 Validar performance em torneios
- **Não fazer otimizações de performance ainda**

### Fase 2: Quick Wins (Depois do treino inicial)
**Tempo estimado**: 2-3 dias
1. ✅ Implementar **Bitboard `all_` Incremental** (1 dia)
2. ✅ Implementar **Cache de Explosion Masks** (1 dia)
3. ✅ Testes de validação (perft, torneios) (1 dia)

**Ganho esperado**: 15-25% de speedup

### Fase 3: Major Optimization (Quando tiver tempo)
**Tempo estimado**: 1-2 semanas
1. ✅ Implementar **Incremental Hash Updates** (3-5 dias)
2. ✅ Testing extensivo (2-3 dias)
3. ✅ Benchmark e validação (1-2 dias)

**Ganho esperado**: 40-60% de speedup total

### Fase 4: Search Tuning (Após NNUE treinado)
1. ✅ Benchmark baseline com NNUE treinado
2. ✅ Testar ajustes de futility pruning (depth 5 vs 6)
3. ✅ Implementar futility move count limit (opcional)
4. ✅ Testar outras adaptações (null move, LMR, extensions)
5. ✅ Validar com torneios (1000+ games vs Fairy-Stockfish)

**Ganho esperado**: 5-15% de playing strength

### Fase 5: Advanced (Futuro distante)
**Tempo estimado**: Várias semanas
1. ⚠️ Reestruturar NNUE updates para capturas
2. ⚠️ Considerar SIMD para operações de bitboard
3. ⚠️ Profile-guided optimization

**Ganho esperado**: 70-100% de speedup total (otimista)

---

## 🧪 Testing Strategy

Para cada otimização, seguir este processo:

### 1. Validação de Correção
```bash
# Perft tests
./seer perft 6

# Tournament vs Fairy-Stockfish
cutechess-cli -engine cmd=./seer.exe -engine cmd=fairy-stockfish.exe \
  -rounds 1000 -variant atomic
```

### 2. Performance Benchmark
```bash
# NPS benchmark em posições típicas
./seer bench

# Comparar antes/depois
echo "Antes: X NPS"
echo "Depois: Y NPS"
echo "Speedup: Y/X"
```

### 3. Regression Testing
- Verificar que não há novos illegal moves
- Comparar resultados de busca (mesma posição deve dar mesmo best move)
- Verificar hashes (mesma posição deve ter mesmo hash)

---

## 📝 Notas de Implementação

### Cuidados Especiais

1. **Consistência de Hash**:
   - Crucial para TT (transposition table)
   - Um único erro pode corromper toda a busca
   - Sempre validar com hash recalculado do zero em debug mode

2. **Bitboard Consistency**:
   - `all_` deve SEMPRE ser igual a união de todos os bitboards de peças
   - Adicionar assertions em debug mode

3. **NNUE Feature Consistency**:
   - Features devem SEMPRE refletir a posição real
   - Bug aqui = NNUE completamente incorreto

### Debug Helpers

Adicionar em debug mode:
```cpp
#ifndef NDEBUG
void board::validate_consistency() {
  // Validate all_ bitboard
  assert(man_.white.all_ == (man_.white.pawn_ | man_.white.knight_ | ...));

  // Validate hash
  zobrist::hash_type recalc_hash = compute_hash_from_scratch();
  assert(man_.white.hash_ == recalc_hash);

  // Validate NNUE features
  // ...
}
#endif
```

---

## 🎯 Objetivos de Performance

### Estado Atual (com resets)
- NPS em posições normais: ~X (medir)
- NPS em posições táticas: ~Y (medir)
- Ratio vs Fairy-Stockfish: ~Z% (medir)

### Meta Fase 2 (Quick Wins)
- +15-25% NPS
- Sem perda de correção

### Meta Fase 3 (Major Optimization)
- +40-60% NPS total
- Competitivo com outros atomic engines

### Meta Final (Long Term)
- +70-100% NPS total
- Entre os engines atomic mais rápidos

---

## 📚 Referências

### Código Similar (para inspiração)
- Stockfish: incremental hash updates
- Lc0: NNUE incremental updates
- Fairy-Stockfish: atomic chess implementation

### Papers/Resources
- [Efficient Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing)
- [Bitboard Techniques](https://www.chessprogramming.org/Bitboards)
- [NNUE Updates](https://github.com/official-stockfish/nnue-pytorch/wiki)

---

## ⚠️ Avisos Importantes

1. **NÃO otimizar antes do treino NNUE estar completo**
   - Prioridade = NNUE funcional e bem treinado
   - Performance vem depois

2. **Sempre validar correção ANTES de performance**
   - Um engine rápido mas incorreto é inútil
   - Perft tests são obrigatórios

3. **Fazer uma otimização de cada vez**
   - Mais fácil de debugar
   - Mais fácil de medir impacto
   - Menos risco de introduzir bugs

4. **Manter versões de backup**
   - Git commit antes de cada otimização
   - Manter binários compilados para comparação

---

## 🔧 Quick Reference: Onde Otimizar

### Arquivos Críticos para Performance
1. **`src/chess/board.cc`** (linhas 1169-1246)
   - `forward_()` - explosões e resets
   - Maior impacto de otimização aqui

2. **`include/chess/board.h`** (linhas 300-310)
   - NNUE feature updates
   - Segunda maior prioridade

3. **`src/chess/board.cc`** (linhas 870-876)
   - `is_legal_()` - verificação de explosões
   - Otimizar explosion_mask lookup

### Ferramentas Úteis
- `perf` (Linux) / `VTune` (Windows): profiling
- `valgrind --tool=callgrind`: hotspot analysis
- `gprof`: function-level profiling

---

**Última atualização**: 2025-12-28
**Status**: 🎯 Foco em treino NNUE, otimizações para depois
