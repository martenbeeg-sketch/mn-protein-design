#!/usr/bin/env python
from __future__ import annotations

import json

import jax
import jax.numpy as jnp


def main() -> None:
    devices = jax.devices()
    payload = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }
    if not devices:
        raise SystemExit("No JAX devices were detected.")
    x = jnp.ones((512, 512), dtype=jnp.float32)
    y = jnp.matmul(x, x).block_until_ready()
    payload["matmul_sum"] = float(jnp.sum(y))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
