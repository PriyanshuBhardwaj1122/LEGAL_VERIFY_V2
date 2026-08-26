"""Verifies the OpenAI strict-structured-output schema transform in
app/providers/llm/openai_llm.py is actually spec-compliant: every object
node gets additionalProperties=False and a required list matching ALL
its properties, and no `default` keyword survives anywhere (OpenAI's
strict mode rejects `default`). This is what makes it structurally
impossible for the model to emit a value from the wrong enum (e.g. a
SourceType value where a QueryIntent is expected) — the bug that caused
the planner's repeated `committee_report` validation failures.

Run: PYTHONPATH=. python tests/test_openai_strict_schema.py
"""
from __future__ import annotations

import sys

from pydantic import BaseModel

from app.providers.llm.openai_llm import _pydantic_to_function_schema

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _assert_strict_compliant(schema, path="root"):
    """Returns a list of violations (empty = compliant)."""
    violations = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            if schema.get("additionalProperties") is not False:
                violations.append(f"{path}: missing additionalProperties=False")
            required = set(schema.get("required", []))
            props = set(schema["properties"].keys())
            if required != props:
                violations.append(f"{path}: required {required} != properties {props}")
        if "default" in schema:
            violations.append(f"{path}: 'default' keyword present (unsupported in strict mode)")
        for k, v in schema.items():
            violations.extend(_assert_strict_compliant(v, f"{path}.{k}"))
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            violations.extend(_assert_strict_compliant(item, f"{path}[{i}]"))
    return violations


class Nested(BaseModel):
    a_required: str
    a_defaulted: int = 5
    a_optional: str | None = None


class Outer(BaseModel):
    name: str
    items: list[Nested] = []
    flag: bool = False


def test_tool_def_has_strict_true():
    print("_pydantic_to_function_schema() — top-level strict flag")
    tool_def = _pydantic_to_function_schema(Outer, "emit_outer")
    check("strict=True is set on the function def", tool_def["function"].get("strict") is True)
    check("tool name is preserved", tool_def["function"]["name"] == "emit_outer")


def test_schema_fully_strict_compliant():
    print("_pydantic_to_function_schema() — nested model, full compliance")
    tool_def = _pydantic_to_function_schema(Outer, "emit_outer")
    schema = tool_def["function"]["parameters"]
    violations = _assert_strict_compliant(schema)
    check("no strict-mode violations anywhere in the schema (incl. nested $defs)", not violations, violations)


def test_defaulted_field_becomes_required_no_default():
    print("_pydantic_to_function_schema() — defaulted fields forced required, default stripped")
    tool_def = _pydantic_to_function_schema(Outer, "emit_outer")
    schema = tool_def["function"]["parameters"]
    check("top-level required includes ALL properties (name, items, flag)", set(schema["required"]) == {"name", "items", "flag"}, schema["required"])

    nested_schema = schema["$defs"]["Nested"]
    check(
        "nested model's required includes ALL its properties too",
        set(nested_schema["required"]) == {"a_required", "a_defaulted", "a_optional"},
        nested_schema["required"],
    )
    check("no 'default' key survives on the defaulted field", "default" not in nested_schema["properties"]["a_defaulted"])


def test_real_node_schemas_are_compliant():
    print("Real node schemas (planner/evaluator/extractor/gap_check) — compliance sweep")
    # Import lazily / individually so a missing optional dep in one node
    # doesn't block checking the others.
    checked = 0
    try:
        from app.graph.nodes.planner import LLMResearchPlan
        tool_def = _pydantic_to_function_schema(LLMResearchPlan, "emit_research_plan")
        violations = _assert_strict_compliant(tool_def["function"]["parameters"])
        check("LLMResearchPlan (planner) schema is strict-compliant", not violations, violations)
        # This is the exact schema that used to let the model put a
        # SourceType value where a QueryIntent belongs — confirm the
        # $ref still points only at QueryIntent's enum (the enum itself
        # was never the bug; enforcement was).
        intent_ref = tool_def["function"]["parameters"]["$defs"]["LLMSubQuery"]["properties"]["intent"]["$ref"]
        check("intent field still refs QueryIntent specifically", intent_ref == "#/$defs/QueryIntent", intent_ref)
        checked += 1
    except ImportError as e:
        print(f"  (skipped planner — {e})")

    try:
        from app.graph.nodes.evaluator import RelevanceBatch
        tool_def = _pydantic_to_function_schema(RelevanceBatch, "emit_relevance_scores")
        violations = _assert_strict_compliant(tool_def["function"]["parameters"])
        check("RelevanceBatch (evaluator) schema is strict-compliant", not violations, violations)
        checked += 1
    except ImportError as e:
        print(f"  (skipped evaluator — {e})")

    try:
        from app.graph.nodes.extractor import LLMEvidenceBatch
        tool_def = _pydantic_to_function_schema(LLMEvidenceBatch, "emit_evidence")
        violations = _assert_strict_compliant(tool_def["function"]["parameters"])
        check("LLMEvidenceBatch (extractor) schema is strict-compliant", not violations, violations)
        checked += 1
    except ImportError as e:
        print(f"  (skipped extractor — {e})")

    try:
        from app.graph.nodes.gap_check import LLMGapBatch
        tool_def = _pydantic_to_function_schema(LLMGapBatch, "emit_gaps")
        violations = _assert_strict_compliant(tool_def["function"]["parameters"])
        check("LLMGapBatch (gap_check) schema is strict-compliant", not violations, violations)
        checked += 1
    except ImportError as e:
        print(f"  (skipped gap_check — {e})")

    check("at least one real node schema was actually checked", checked > 0, checked)


def main():
    test_tool_def_has_strict_true()
    test_schema_fully_strict_compliant()
    test_defaulted_field_becomes_required_no_default()
    test_real_node_schemas_are_compliant()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
