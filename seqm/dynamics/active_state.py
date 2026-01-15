import torch


def active_state_tensor(active_state, nmol: int, device) -> torch.Tensor:
    """
    Convert an active_state value (int or tensor) to a 1D tensor of length nmol.
    active_state is expected to be 1-based for excited states, 0 for ground.
    """
    if torch.is_tensor(active_state):
        t = active_state.to(device)
        if t.dim() == 0:
            t = t.expand(nmol)
        return t.long()
    val = int(active_state)
    return torch.full((nmol,), val, device=device, dtype=torch.long)
