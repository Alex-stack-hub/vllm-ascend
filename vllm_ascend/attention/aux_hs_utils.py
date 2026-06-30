"""Helpers for flatten/unpack aux hidden states in graph capture."""
from typing import Any

import torch


def flatten_aux_hidden_states_output(output: Any) -> Any:
    """If output is (Tensor, list[Tensor]), flatten to (Tensor, *list).

    Only matches the exact shape: (torch.Tensor, list/tuple[torch.Tensor]).
    Everything else is passed through unchanged.
    """
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


def unpack_aux_hidden_states_output(
    hidden_states: Any,
) -> tuple[Any, list[Any] | None]:
    """Return (hs, aux_list_or_None).

    * Eager format:  (Tensor, list/tuple[Tensor]) → (hs, [aux0, ...])
    * Graph flat:    flat tuple of Tensors        → (hs, [aux0, ...])
    * Single tensor:                              → (hs, None)
    * Anything else: raises ValueError.
    """
    if not isinstance(hidden_states, tuple):
        return hidden_states, None

    n = len(hidden_states)
    if n == 1:
        return hidden_states[0], None

    if n == 2 and isinstance(hidden_states[1], (list, tuple)):
        hs, aux = hidden_states
        if not torch.is_tensor(hs):
            raise ValueError(
                f"Expected (Tensor, list[Tensor]) for aux hidden states, "
                f"got ({type(hs).__name__}, {type(aux).__name__})"
            )
        if not aux or not all(torch.is_tensor(t) for t in aux):
            raise ValueError(
                f"Expected list of Tensors in aux position, "
                f"got {[type(t).__name__ for t in aux]}"
            )
        return hs, list(aux)

    if n > 1 and all(torch.is_tensor(t) for t in hidden_states):
        return hidden_states[0], list(hidden_states[1:])

    raise ValueError(
        f"Unexpected aux hidden states output format: "
        f"tuple len={n}, types={[type(t).__name__ for t in hidden_states]}"
    )
