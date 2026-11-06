import torch


class KVCache:
    """Stores projected keys and values for each transformer layer."""

    def __init__(
        self,
        num_layers: int = 12,
        batch_size: int = 1,
        num_heads: int = 12,
        head_dim: int = 64,
        max_length: int = 512,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(num_layers, batch_size, num_heads, head_dim, max_length) <= 0:
            raise ValueError("Cache dimensions must be positive")

        self.max_length = max_length
        self.position = 0
        self.key_cache = [
            torch.zeros(
                batch_size, num_heads, max_length, head_dim,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(
                batch_size, num_heads, max_length, head_dim,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Invalid KV cache layer index: {layer_idx}")
        if key.shape != value.shape or key.dim() != 4:
            raise ValueError("key and value must have the same four-dimensional shape")

        end = self.position + key.size(2)
        if end > self.max_length:
            raise ValueError(
                f"KV cache capacity exceeded: requested {end}, "
                f"maximum is {self.max_length}"
            )

        expected_shape = self.key_cache[layer_idx].shape
        if (
            key.size(0) != expected_shape[0]
            or key.size(1) != expected_shape[1]
            or key.size(3) != expected_shape[3]
        ):
            raise ValueError("key/value dimensions do not match the cache")

        self.key_cache[layer_idx][:, :, self.position:end, :].copy_(key)
        self.value_cache[layer_idx][:, :, self.position:end, :].copy_(value)
        
        return (
            self.key_cache[layer_idx][:, :, :end, :],
            self.value_cache[layer_idx][:, :, :end, :],
        )

    def advance(self, seq_length: int = 1) -> None:
        if seq_length < 0 or self.position + seq_length > self.max_length:
            raise ValueError("Invalid KV cache position")
        self.position += seq_length

    def reset(self) -> None:
        self.position = 0