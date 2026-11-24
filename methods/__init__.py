"""Damage-mapping methods. Each sub-package exposes `Method(cfg, store)` with
`damage(field_id, night) -> {score, valid, meta, layers}`."""
import importlib


def load(name):
    return importlib.import_module(f"methods.{name}")
