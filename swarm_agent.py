"""CJR Swarm Orchestrator — Main autonomous pipeline loop."""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import traceback
from datetime import datetime, timezone
from typing import Final

from agents import ImplementerAgent, PlannerAgent, QAReviewerAgent
from git_ops import GitController
from schemas import (
    ActionState,
    CodePayload,
    CommitInfo,
    PipelineStage,
    SwarmConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: Final[str] = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger("swarm_agent")

# ---------------------------------------------------------------------------
# Feature objectives — curated list of complex backend projects
# ---------------------------------------------------------------------------

FEATURE_OBJECTIVES: Final[list[str]] = [
    # 1
    (
        "Build a distributed rate limiter service using token bucket algorithm "
        "with Redis-backed state, supporting sliding window counters, burst "
        "allowances, and per-tenant quota management via REST API"
    ),
    # 2
    (
        "Implement a real-time event sourcing engine with CQRS pattern, "
        "supporting snapshotting, event replay, projection rebuilding, and "
        "saga orchestration for distributed transactions"
    ),
    # 3
    (
        "Create an async task queue system with priority scheduling, dead letter "
        "queues, exponential backoff retries, task chaining, cron scheduling, "
        "and observability hooks"
    ),
    # 4
    (
        "Design a multi-tenant feature flag service with percentage rollouts, "
        "A/B testing variants, user segment targeting, audit logging, and a "
        "typed SDK generator"
    ),
    # 5
    (
        "Build a schema migration orchestrator supporting forward/backward "
        "migrations, dry-run previews, dependency-ordered execution, rollback "
        "plans, and multi-database dialect support"
    ),
    # 6
    (
        "Implement a GraphQL federation gateway with schema stitching, "
        "automatic type merging, subscription forwarding, query planning, "
        "and per-subgraph authentication middleware"
    ),
    # 7
    (
        "Design a plugin architecture system with hot-reloading, dependency "
        "resolution via DAG, sandboxed execution, lifecycle hooks, typed "
        "extension points, and a plugin marketplace registry API"
    ),
    # 8
    (
        "Build an ETL pipeline framework with configurable DAG execution, "
        "incremental extraction, schema evolution handling, data quality "
        "assertions, lineage tracking, and partition-aware backfills"
    ),
    # 9
    (
        "Create a WebSocket pub/sub broker supporting topic hierarchies, "
        "wildcard subscriptions, message persistence with replay, presence "
        "tracking, binary frame multiplexing, and horizontal scaling via "
        "Redis Streams"
    ),
    # 10
    (
        "Implement a secrets management vault with envelope encryption, "
        "automatic rotation policies, access audit trails, dynamic database "
        "credential leasing, transit encryption API, and HSM integration "
        "abstractions"
    ),
    # 11
    (
        "Build an API gateway with circuit breaker pattern, adaptive rate "
        "limiting, request/response transformation pipelines, canary routing, "
        "JWT validation, and a declarative route configuration DSL"
    ),
    # 12
    (
        "Design a distributed tracing collector compatible with OpenTelemetry "
        "OTLP, supporting tail-based sampling, trace-to-log correlation, "
        "service dependency map generation, anomaly detection, and Parquet "
        "cold-storage export"
    ),
    # 13
    (
        "Create a log aggregation pipeline with structured log parsing, "
        "multi-source ingestion (syslog, journald, Kafka), real-time "
        "pattern-based alerting, retention policies, and a query language "
        "with full-text search over compressed archives"
    ),
    # 14
    (
        "Build a config management service with typed schemas, environment "
        "inheritance, live push notifications via SSE, diff-based audit log, "
        "rollback to any historical version, and encrypted secret fields"
    ),
    # 15
    (
        "Implement an RBAC authorization engine with hierarchical roles, "
        "attribute-based policy rules (ABAC hybrid), resource-scoped "
        "permissions, policy simulation/dry-run endpoint, and decision "
        "caching with TTL invalidation"
    ),
    # 16
    (
        "Design a webhook delivery system with at-least-once guarantees, "
        "HMAC signature verification, exponential retry with jitter, "
        "delivery status dashboard, payload transformation templates, "
        "and IP-allow-list enforcement"
    ),
    # 17
    (
        "Build a data validation pipeline supporting JSON Schema, Avro, "
        "and Protobuf inputs, with pluggable custom rule engines, "
        "quarantine queues for invalid records, validation report generation, "
        "and schema registry integration"
    ),
    # 18
    (
        "Create a cache invalidation system with tag-based purging, "
        "hierarchical key namespaces, write-through and write-behind "
        "strategies, stale-while-revalidate semantics, and distributed "
        "cache coherence via pub/sub"
    ),
    # 19
    (
        "Implement a health check orchestrator supporting HTTP, TCP, gRPC, "
        "and custom script probes, with dependency-aware status aggregation, "
        "configurable thresholds, incident auto-escalation, status page "
        "generation, and Prometheus metric export"
    ),
    # 20
    (
        "Build a deployment pipeline manager with blue/green and canary "
        "strategies, automated rollback on SLO breach, approval gates, "
        "environment promotion chains, artifact versioning, and Slack/Teams "
        "notification hooks"
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert a human-readable project name to a GitHub-safe repo slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "swarm-project"


_BANNER: Final[str] = r"""
 ██████╗     ██╗██████╗     ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗
██╔════╝     ██║██╔══██╗    ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║
██║          ██║██████╔╝    ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║
██║     ██   ██║██╔══██╗    ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║
╚██████╗╚█████╔╝██║  ██║    ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
 ╚═════╝ ╚════╝ ╚═╝  ╚═╝    ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
              O R C H E S T R A T O R   v1.0.0
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> ActionState:
    """Execute the full autonomous swarm pipeline.

    Stages:
        1. **Planning** — generate an implementation plan for a random objective.
        2. **Implementation** — produce code files from the plan.
        3. **QA Review** — review code; retry implementation on failure.
        4. **Git Push** — commit and push code to a GitHub repository.

    Returns
    -------
    ActionState
        The final pipeline state with artefacts attached.
    """

    # ── 0. Bootstrap ──────────────────────────────────────────────────
    config = SwarmConfig.from_env()
    objective = random.choice(FEATURE_OBJECTIVES)  # noqa: S311
    state = ActionState(
        feature_objective=objective,
    )

    logger.info("🚀 Starting pipeline for: %s", objective)

    # ── 1. PLANNING ───────────────────────────────────────────────────
    try:
        state.current_stage = PipelineStage.PLANNING
        logger.info("📋 [PLANNING] Generating implementation plan…")

        planner = PlannerAgent(config)
        plan = planner.generate_plan(objective)
        state.plan = plan

        logger.info(
            "📋 Plan generated: %s with %d tasks",
            plan.project_name,
            len(plan.tasks),
        )
    except Exception as exc:
        logger.error("❌ Planning stage failed: %s", exc)
        state.log_error(f"Planning failed: {exc}")
        return state

    # ── 2. IMPLEMENTATION ─────────────────────────────────────────────
    try:
        state.current_stage = PipelineStage.IMPLEMENTATION
        logger.info("💻 [IMPLEMENTATION] Generating code from plan…")

        implementer = ImplementerAgent(config)
        code = implementer.implement(plan)
        state.code = code

        logger.info("💻 Code generated: %d files", len(code.files))
    except Exception as exc:
        logger.error("❌ Implementation stage failed: %s", exc)
        state.log_error(f"Implementation failed: {exc}")
        return state

    # ── 3. QA REVIEW (with retry loop) ────────────────────────────────
    max_retries: int = config.max_retries if hasattr(config, "max_retries") else 3
    try:
        state.current_stage = PipelineStage.QA_REVIEW
        logger.info("🔍 [QA REVIEW] Reviewing generated code…")

        reviewer = QAReviewerAgent(config)
        qa_report = reviewer.review(plan, code)
        state.qa_report = qa_report

        retries = 0
        while qa_report.retry_required and retries < max_retries:
            retries += 1
            state.retries = retries
            logger.warning(
                "🔄 QA flagged issues (attempt %d/%d) — re-implementing with feedback…",
                retries,
                max_retries,
            )
            logger.info(
                "   Recommendations: %s",
                "; ".join(qa_report.recommendations[:5]),
            )

            # Re-implement with QA feedback
            code = implementer.implement(plan, feedback=qa_report.recommendations)
            state.code = code
            logger.info("💻 Re-generated: %d files", len(code.files))

            # Re-review
            qa_report = reviewer.review(plan, code)
            state.qa_report = qa_report

        if qa_report.retry_required:
            logger.warning(
                "⚠️  QA still flagging issues after %d retries — proceeding anyway",
                max_retries,
            )

        logger.info(
            "✅ QA completed with %.1f%% pass rate",
            qa_report.pass_rate * 100.0,
        )
    except Exception as exc:
        logger.error("❌ QA review stage failed: %s", exc)
        state.log_error(f"QA review failed: {exc}")
        return state

    # ── 4. GIT PUSH ───────────────────────────────────────────────────
    try:
        state.current_stage = PipelineStage.GIT_PUSH
        logger.info("📤 [GIT PUSH] Pushing code to GitHub…")

        git_controller = GitController(config)
        repo_name = _slugify(plan.project_name)
        file_paths = [f.filepath for f in code.files]

        commit_info = CommitInfo(
            message=git_controller._generate_commit_message(  # noqa: SLF001
                plan.project_name, file_paths
            ),
            files_changed=file_paths,
            branch="main",
        )

        result = git_controller.push_code(code, repo_name, commit_info)
        state.git_result = result

        if result.success:
            logger.info(
                "🚀 Pushed to %s (commit: %s)",
                result.remote_url,
                (result.commit_sha or "")[:8],
            )
        else:
            logger.error(
                "❌ Git push failed: %s", result.error_message
            )
    except Exception as exc:
        logger.error("❌ Git push stage failed: %s", exc)
        state.log_error(f"Git push failed: {exc}")
        return state

    # ── 5. COMPLETE ───────────────────────────────────────────────────
    state.mark_complete()
    elapsed = (state.completed_at - state.started_at).total_seconds()  # type: ignore[operator]

    logger.info("=" * 72)
    logger.info("🏁 PIPELINE COMPLETE")
    logger.info("   Objective  : %s", objective[:80])
    logger.info("   Project    : %s", plan.project_name)
    logger.info("   Files      : %d", len(code.files))
    logger.info("   QA Pass    : %.1f%%", qa_report.pass_rate * 100.0)
    logger.info("   Retries    : %d", state.retries)
    logger.info("   Git Push   : %s", "✅" if (state.git_result and state.git_result.success) else "❌")
    logger.info("   Duration   : %.1fs", elapsed)
    logger.info("=" * 72)

    return state


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint — run the full CJR Swarm Orchestrator pipeline."""
    logger.info(_BANNER)
    logger.info("🕐 Pipeline starting at %s", datetime.now(tz=timezone.utc).isoformat())

    try:
        state = run_pipeline()

        if state.current_stage != PipelineStage.DONE:
            logger.error("💀 Pipeline finished with errors: %s", state.error_log)
            sys.exit(1)

        logger.info("🎉 CJR Swarm Orchestrator finished successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception:
        logger.critical("💀 Unhandled exception in pipeline:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
