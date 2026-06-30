
"""Helpers for flatten/unpack aux hidden states in graph capture."""
from typing import Any
import torch

def flatten_aux_hidden_states_output(output: Any) -> Any:
    """If output is (Tensor, list[Tensor]), flatten to (Tensor, *list)."""
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    hs, aux = output
    if not torch.is_tensor(hs):
        return output
    if not isinstance(aux, (list, tuple)):
        return output
    if not aux or not all(torch.is_tensor(t) for t in aux):
        return output
    return (hs, *aux)


def unpack_aux_hidden_states_output(hidden_states: Any):
    """Return (hs, aux_list_or_None).
    Handles both eager (hs, [aux0,aux1,aux2]) and graph flat (hs, aux0, aux1, aux2).
    """
    if not isinstance(hidden_states, tuple):
        return hidden_states, None
    n = len(hidden_states)
    if n == 1:
        return hidden_states[0], None
    if n == 2 and isinstance(hidden_states[1], (list, tuple)):
        return hidden_states[0], list(hidden_states[1])
    if n > 1 and all(torch.is_tensor(t) for t in hidden_states):
        return hidden_states[0], list(hidden_states[1:])
    return hidden_states[0] if n == 1 else hidden_states, None
