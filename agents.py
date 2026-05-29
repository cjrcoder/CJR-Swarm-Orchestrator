"""Multi-agent swarm brain — Planner, Implementer, and QA Reviewer.

Each agent wraps a specific LLM endpoint and enforces typed I/O
boundaries via Pydantic schemas.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any

import openai
from google import genai

from schemas import (
    CodePayload,
    ImplementationPlan,
    QAReport,
    SwarmConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response, handling markdown code fences.

    Supports responses wrapped in ``json ... `` or bare JSON objects.
    Raises ``ValueError`` when no valid JSON can be recovered.
    """
    # 1. Strip markdown code fences (```json … ``` or ``` … ```)
    stripped = re.sub(
        r"```(?:json)?\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    stripped = stripped.rstrip("`").strip()

    # 2. First attempt – direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Fallback – locate outermost { … } pair
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = stripped[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response "
        f"(first 200 chars): {text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Planner Agent  (Gemini)
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Uses NVIDIA Llama-3 to decompose feature objectives into atomic implementation plans."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key.get_secret_value(),
        )
        self.model_name = "meta/llama-3.3-70b-instruct"

    def generate_plan(self, feature_objective: str) -> ImplementationPlan:
        """Generate a structured implementation plan for *feature_objective*.

        Returns a fully-validated :class:`ImplementationPlan` instance.
        """
        schema_hint = textwrap.dedent("""\
            {
              "plan_id": "<uuid string>",
              "feature_objective": "<string>",
              "architecture_notes": "<string>",
              "tech_stack": ["<string>", ...],
              "tasks": [
                {
                  "task_id": "<uuid string>",
                  "title": "<short title>",
                  "description": "<detailed description>",
                  "acceptance_criteria": ["<criterion>", ...],
                  "priority": <1-5 integer, 1=highest>,
                  "estimated_complexity": "low | medium | high"
                }
              ]
            }
        """)

        system_prompt = textwrap.dedent(f"""\
            You are a senior software architect tasked with creating
            implementation plans for software features.

            RULES:
            1. Decompose the feature objective into small, atomic tasks.
            2. Each task must be independently implementable and testable.
            3. Provide architecture notes and a tech stack recommendation.
            4. Return ONLY strict JSON — no commentary, no markdown.
            5. The JSON MUST match this exact schema:

            {schema_hint}

            Generate unique UUID-v4 strings for plan_id and each task_id.
        """)

        user_prompt = (
            f"Create a detailed implementation plan for the following "
            f"feature objective:\n\n{feature_objective}"
        )

        logger.info(
            "PlannerAgent: requesting plan for objective (%.80s…)",
            feature_objective,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw_text = completion.choices[0].message.content or ""
            logger.debug("PlannerAgent raw response: %.500s", raw_text)

            data = _extract_json(raw_text)
            plan = ImplementationPlan.model_validate(data)
            logger.info(
                "PlannerAgent: plan '%s' created with %d task(s)",
                plan.project_name,
                len(plan.tasks),
            )
            return plan

        except Exception:
            logger.exception("PlannerAgent: failed to generate plan")
            raise


# ---------------------------------------------------------------------------
# Implementer Agent  (NVIDIA DeepSeek-Coder)
# ---------------------------------------------------------------------------

class ImplementerAgent:
    """Uses NVIDIA DeepSeek-Coder to generate production Python code from a typed plan."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key.get_secret_value(),
        )

    def implement(
        self,
        plan: ImplementationPlan,
        feedback: list[str] | None = None,
    ) -> CodePayload:
        """Generate production Python code that satisfies *plan*.

        Returns a validated :class:`CodePayload` instance.
        """
        schema_hint = textwrap.dedent("""\
            {
              "payload_id": "<uuid string>",
              "plan_id": "<the plan_id you received>",
              "files": [
                {
                  "filepath": "<relative/path.py>",
                  "language": "python",
                  "content": "<full file source code as a string>",
                  "description": "<what this file does>"
                }
              ],
              "entry_point": "<main entry point file>",
              "dependencies": ["<pip package>", ...]
            }
        """)

        tasks_block = "\n".join(
            f"  {i + 1}. [complexity={t.estimated_complexity}/10] {t.title}\n"
            f"     {t.description}\n"
            f"     Acceptance: {', '.join(t.acceptance_criteria)}"
            for i, t in enumerate(plan.tasks)
        )

        system_prompt = textwrap.dedent(f"""\
            You are a senior Python developer. You receive an implementation
            plan and produce production-quality Python code.

            RULES:
            1. Follow PEP 8.  Use type hints everywhere.
            2. Write clean, well-documented, fully working code.
            3. Return ONLY strict JSON — no markdown, no commentary.
            4. The JSON MUST match this exact schema:

            {schema_hint}

            Generate a unique UUID-v4 for payload_id.
            Set plan_id to the value provided in the plan.
        """)

        user_prompt = textwrap.dedent(f"""\
            IMPLEMENTATION PLAN
            ====================
            Project       : {plan.project_name}
            Objective     : {plan.feature_objective}
            Architecture  : {plan.architecture_notes}
            Tech Stack    : {', '.join(plan.tech_stack)}

            TASKS
            -----
            {tasks_block}

            Generate production Python code that fulfils every task above.
        """)

        if feedback:
            feedback_block = "\n".join(f"- {f}" for f in feedback)
            user_prompt += textwrap.dedent(f"""

            QA FEEDBACK (address these issues):
            {feedback_block}
            """)

        logger.info(
            "ImplementerAgent: generating code for plan '%s'",
            plan.project_name,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self._config.nvidia_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            raw_text = completion.choices[0].message.content or ""
            logger.debug("ImplementerAgent raw response: %.500s", raw_text)

            data = _extract_json(raw_text)

            # Ensure plan_id linkage — use first task's id as a stable reference
            if plan.tasks:
                data["plan_id"] = str(plan.tasks[0].id)

            payload = CodePayload.model_validate(data)
            logger.info(
                "ImplementerAgent: generated %d file(s) for '%s'",
                len(payload.files),
                plan.project_name,
            )
            return payload

        except Exception:
            logger.exception("ImplementerAgent: failed to generate code")
            raise


# ---------------------------------------------------------------------------
# QA Reviewer Agent  (NVIDIA Llama-3)
# ---------------------------------------------------------------------------

class QAReviewerAgent:
    """Uses NVIDIA Llama-3 to review code, generate tests, and determine quality."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key.get_secret_value(),
        )

    def review(
        self,
        plan: ImplementationPlan,
        code: CodePayload,
    ) -> QAReport:
        """Review *code* against *plan* and produce a QA report.

        Returns a validated :class:`QAReport` instance.
        """
        schema_hint = textwrap.dedent("""\
            {
              "report_id": "<uuid string>",
              "payload_id": "<the payload_id from the code payload>",
              "overall_quality": <integer 1-10>,
              "issues": [
                {
                  "severity": "critical | major | minor | info",
                  "location": "<file:line or description>",
                  "message": "<what is wrong>",
                  "suggestion": "<how to fix>"
                }
              ],
              "test_cases": [
                {
                  "name": "<test function name>",
                  "description": "<what it verifies>",
                  "code": "<full pytest test code>"
                }
              ],
              "approved": <true | false>,
              "summary": "<executive summary of the review>"
            }
        """)

        files_block = "\n\n".join(
            f"--- {f.filepath} ---\n{f.content}"
            for f in code.files
        )

        tasks_block = "\n".join(
            f"  - {t.title}: {t.description}"
            for t in plan.tasks
        )

        system_prompt = textwrap.dedent(f"""\
            You are a senior QA engineer and code reviewer.  You receive an
            implementation plan and generated code, then produce a thorough
            quality report.

            RULES:
            1. Evaluate correctness, style, security, performance, and test
               coverage.
            2. Generate pytest-compatible test cases for critical paths.
            3. Set "approved" to true ONLY if overall_quality >= 7 and there
               are zero critical issues.
            4. Return ONLY strict JSON — no markdown, no commentary.
            5. The JSON MUST match this schema:

            {schema_hint}

            Generate a unique UUID-v4 for report_id.
            Set payload_id to the value from the code payload.
        """)

        user_prompt = textwrap.dedent(f"""\
            IMPLEMENTATION PLAN
            ====================
            Objective : {plan.feature_objective}
            Tasks:
            {tasks_block}

            GENERATED CODE
            ===============
            {files_block}

            Review the code above against the plan and produce a QA report.
        """)

        logger.info(
            "QAReviewerAgent: reviewing code for plan_id='%s'",
            code.plan_id,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self._config.llama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            raw_text = completion.choices[0].message.content or ""
            logger.debug("QAReviewerAgent raw response: %.500s", raw_text)

            data = _extract_json(raw_text)

            # Ensure payload_id linkage — use the code's plan_id
            data["payload_id"] = str(code.plan_id)

            report = QAReport.model_validate(data)
            logger.info(
                "QAReviewerAgent: overall_passed=%s, retry_required=%s",
                report.overall_passed,
                report.retry_required,
            )
            return report

        except Exception:
            logger.exception("QAReviewerAgent: failed to produce QA report")
            raise
