"""Strict JSON schemas for AgentDS ultrasound and morphology descriptors."""

import json
from typing import Dict, Iterable, List, Tuple


class DescriptorSchemaError(ValueError):
    """Raised when a descriptor response violates its required JSON schema."""


ULTRASOUND_ENUMS = {
    "echo_pattern": {
        "anechoic",
        "hypoechoic",
        "isoechoic",
        "hyperechoic",
        "mixed_solid_and_cystic",
    },
    "internal_structure": {"homogeneous", "heterogeneous"},
    "posterior_features": {
        "no_posterior_features",
        "enhancement",
        "shadowing",
    },
}
CALCIFICATION_VALUES = {
    "absent",
    "macrocalcifications",
    "microcalcifications",
}
MORPHOLOGY_ENUMS = {
    "shape": {"round", "oval", "irregular"},
    "orientation": {"parallel", "non_parallel"},
    "margin": {
        "circumscribed",
        "indistinct",
        "angular",
        "microlobulated",
        "spiculated",
    },
}


def _reject_duplicate_keys(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DescriptorSchemaError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_exact_json_object(text: str, schema_name: str) -> Dict[str, object]:
    if not isinstance(text, str) or not text.strip():
        raise DescriptorSchemaError(f"{schema_name}: response is empty")
    try:
        value = json.loads(text.strip(), object_pairs_hook=_reject_duplicate_keys)
    except DescriptorSchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise DescriptorSchemaError(
            f"{schema_name}: response must contain only one complete JSON object: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise DescriptorSchemaError(f"{schema_name}: top-level JSON value must be an object")
    return value


def _require_exact_keys(
    value: Dict[str, object],
    expected_keys: Iterable[str],
    schema_name: str,
) -> None:
    expected = set(expected_keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DescriptorSchemaError(
            f"{schema_name}: key mismatch; missing={missing}, extra={extra}"
        )


def _require_enum_string(
    value: Dict[str, object],
    key: str,
    allowed: Iterable[str],
    schema_name: str,
) -> None:
    field_value = value[key]
    allowed_values = set(allowed)
    if not isinstance(field_value, str):
        raise DescriptorSchemaError(f"{schema_name}.{key}: value must be a string")
    if field_value not in allowed_values:
        raise DescriptorSchemaError(
            f"{schema_name}.{key}: invalid value {field_value!r}; "
            f"allowed={sorted(allowed_values)}"
        )


def parse_ultrasound_descriptors(text: str) -> Dict[str, object]:
    schema_name = "ultrasound_descriptors"
    value = _parse_exact_json_object(text, schema_name)
    _require_exact_keys(
        value,
        (*ULTRASOUND_ENUMS.keys(), "calcification"),
        schema_name,
    )
    for key, allowed in ULTRASOUND_ENUMS.items():
        _require_enum_string(value, key, allowed, schema_name)

    calcification = value["calcification"]
    if not isinstance(calcification, list) or not calcification:
        raise DescriptorSchemaError(
            f"{schema_name}.calcification: value must be a non-empty JSON array"
        )
    if any(not isinstance(item, str) for item in calcification):
        raise DescriptorSchemaError(
            f"{schema_name}.calcification: every array item must be a string"
        )
    invalid = sorted(set(calcification) - CALCIFICATION_VALUES)
    if invalid:
        raise DescriptorSchemaError(
            f"{schema_name}.calcification: invalid values={invalid}; "
            f"allowed={sorted(CALCIFICATION_VALUES)}"
        )
    if len(calcification) != len(set(calcification)):
        raise DescriptorSchemaError(
            f"{schema_name}.calcification: duplicate values are not allowed"
        )
    if "absent" in calcification and len(calcification) != 1:
        raise DescriptorSchemaError(
            f"{schema_name}.calcification: 'absent' cannot coexist with calcifications"
        )
    return value


def parse_morphology_descriptors(text: str) -> Dict[str, object]:
    schema_name = "morphology_descriptors"
    value = _parse_exact_json_object(text, schema_name)
    _require_exact_keys(value, MORPHOLOGY_ENUMS.keys(), schema_name)
    for key, allowed in MORPHOLOGY_ENUMS.items():
        _require_enum_string(value, key, allowed, schema_name)
    return value
