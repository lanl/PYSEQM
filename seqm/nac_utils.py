from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class NACConfig:
    enabled: bool
    pairs: List[Tuple[int, int]]


def _dedup_pairs(pairs: Sequence[Tuple[int, int]], nroots: Optional[int]) -> List[Tuple[int, int]]:
    seen = set()
    cleaned: List[Tuple[int, int]] = []
    for s1, s2 in pairs:
        if s1 == s2:
            continue
        a, b = sorted((int(s1), int(s2)))
        if nroots is not None:
            if not (1 <= a <= nroots and 1 <= b <= nroots):
                continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def resolve_nac_config(
    seqm_parameters: Dict,
    *,
    nroots: Optional[int] = None,
    default_enabled: bool = False,
) -> NACConfig:
    """
    Unified NAC configuration:
      - seqm_parameters['nonadiabatic']['compute_nac']: on/off
      - seqm_parameters['nonadiabatic']['states']: list of states; all unique pairs are used
      - seqm_parameters['nonadiabatic']['pairs']: explicit list of (i, j) pairs

    Legacy inputs like cis_nac or nonadiabatic_coupling are intentionally ignored.
    """
    na_cfg = dict(seqm_parameters.get("nonadiabatic", {}))
    enabled = bool(na_cfg.get("compute_nac", default_enabled))
    pairs: List[Tuple[int, int]] = []

    state_list = na_cfg.get("states")
    if state_list:
        uniq_states = sorted({int(s) for s in state_list if isinstance(s, (int, float))})
        pairs.extend((uniq_states[i], uniq_states[j]) for i in range(len(uniq_states)) for j in range(i + 1, len(uniq_states)))

    explicit_pairs = na_cfg.get("pairs")
    if explicit_pairs:
        for st in explicit_pairs:
            if isinstance(st, (list, tuple)) and len(st) == 2:
                pairs.append((int(st[0]), int(st[1])))

    if enabled and not pairs and nroots:
        pairs = [(i, j) for i in range(1, nroots + 1) for j in range(i + 1, nroots + 1)]

    pairs = _dedup_pairs(pairs, nroots)
    return NACConfig(enabled=enabled, pairs=pairs)
