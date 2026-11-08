from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from autobench import Variant
from kedi.forms import Form, FormError

FORM_FACTOR_PREFIX = "kedi.form."


def variant_forms(variant: Variant) -> dict[str, Form]:
    """The Kedi forms a variant selects with ``kedi.form.<group>`` factors.

    A factor's value is the form's Kedi text, or an object with ``text`` and the optional
    ``parent`` fingerprint and ``note``. Pass the result to ``compile_program(forms=...)``.
    """
    forms: dict[str, Form] = {}
    for factor in variant.factors:
        if factor.name.startswith(FORM_FACTOR_PREFIX):
            group = factor.name.removeprefix(FORM_FACTOR_PREFIX)
            forms[group] = _form(factor.name, factor.value)
    return forms


def _form(name: str, value: object) -> Form:
    if isinstance(value, str):
        return Form(value, origin=f"<factor {name}>")
    if not isinstance(value, Mapping):
        raise FormError(f"Factor {name} must be form text or an object with 'text'")
    fields = cast(Mapping[str, object], value)
    text = fields.get("text")
    if not isinstance(text, str):
        raise FormError(f"Factor {name} must be form text or an object with 'text'")
    unknown = sorted(set(fields) - {"text", "parent", "note"})
    if unknown:
        raise FormError(f"Factor {name} has unknown fields {', '.join(unknown)}")
    parent, note = fields.get("parent"), fields.get("note")
    if not isinstance(parent, str | None) or not isinstance(note, str | None):
        raise FormError(f"Factor {name}: 'parent' and 'note' must be strings")
    return Form(text, parent, note, f"<factor {name}>")


__all__ = ("FORM_FACTOR_PREFIX", "variant_forms")
