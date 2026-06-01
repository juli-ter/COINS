from __future__ import annotations

def coins_colours() -> dict[str, tuple[float, float, float] | list[tuple[float, float, float]]]:
    return {
        "stable": (0.0, 0.0, 0.8),
        "volatile": (0.8, 0.0, 0.0),
        "lowNoise": (0.1, 0.1, 0.1),
        "medNoise": (0.3, 0.3, 0.3),
        "highNoise": (0.5, 0.5, 0.5),
        "sessions": [
            (0.5, 0.5, 1.0),
            (0.25, 0.25, 0.75),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.25),
        ],
    }
