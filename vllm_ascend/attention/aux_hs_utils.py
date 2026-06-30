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


def pack_aux_for_graph(output: Any) -> Any:
    """Pack flat (hs, aux0, aux1, aux2) into (hs, stacked_aux).

    NPU graph capture may not handle tuples with many elements.
    A 2-tuple is reliably captured and replayed.
    """
    if not isinstance(output, tuple) or len(output) <= 2:
        return output
    if not torch.is_tensor(output[0]):
        return output
    if not all(torch.is_tensor(t) for t in output[1:]):
        return output
    stacked = torch.cat([t.unsqueeze(0) for t in output[1:]], dim=0)
    return (output[0], stacked)


def unpack_aux_hidden_states_output(
    hidden_states: Any,
) -> tuple[Any, list[Any] | None]:
    """Return (hs, aux_list_or_None).

    Handles:
    * Packed graph:  (hs, stacked_aux)              -> (hs, [aux0, ...])
    * Eager nested:  (hs, list/tuple[Tensor])        -> (hs, [aux0, ...])
    * Graph flat:    flat tuple of Tensors (>2)      -> (hs, [aux0, ...])
    * Single tensor                                   -> (hs, None)
    """
    if not isinstance(hidden_states, tuple):
        return hidden_states, None

    n = len(hidden_states)
    if n == 1:
        return hidden_states[0], None

    # Packed format: 2-tuple where 2nd element is a stacked tensor
    if n == 2 and torch.is_tensor(hidden_states[1]) and hidden_states[1].dim() >= 2:
        hs, stacked = hidden_states
        return hs, [stacked[i] for i in range(stacked.shape[0])]

    # Eager nested: 2-tuple where 2nd is list/tuple
    if n == 2 and isinstance(hidden_states[1], (list, tuple)):
        hs, aux = hidden_states
        if not torch.is_tensor(hs):
            raise ValueError(f"Expected (Tensor, list), got ({type(hs).__name__}, {type(aux).__name__})")
        if not aux or not all(torch.is_tensor(t) for t in aux):
            raise ValueError(f"Expected list of Tensors, got {[type(t).__name__ for t in aux]}")
        return hs, list(aux)

    # Graph flat: 3+ tensors
    if n > 2 and all(torch.is_tensor(t) for t in hidden_states):
        return hidden_states[0], list(hidden_states[1:])

    raise ValueError(
        f"Unexpected aux hidden states format: tuple len={n}, "
        f"types={[type(t).__name__ for t in hidden_states]}"
    )
