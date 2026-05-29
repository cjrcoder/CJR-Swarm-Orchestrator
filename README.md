# CJR-Swarm-Orchestrator

[![CI](https://github.com/CJRCODER/CJR-Swarm-Orchestrator/actions/workflows/agent_loop.yml/badge.svg)](https://github.com/CJRCODER/CJR-Swarm-Orchestrator/actions/workflows/agent_loop.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-e92063.svg)](https://docs.pydantic.dev/latest/)

A production-grade **multi-agent swarm orchestration framework** that coordinates four specialized AI agents through a sequential pipeline to autonomously plan, implement, review, and deploy code.

---

## Architecture Overview

CJR-Swarm-Orchestrator operates as a four-stage pipeline, where each stage is owned by a dedicated agent role:

```
┌─────────────┐    ┌───────────────┐    ┌───────────────┐    ┌────────────────┐
│   PLANNER   │───▶│ IMPLEMENTER   │───▶│  QA_REVIEWER  │───▶│ GIT_CONTROLLER │
│  (Gemini)   │    │ (DeepSeek/    │    │ (Llama-3 /    │    │   (GitPython)  │
│             │    │  NVIDIA API)  │    │  NVIDIA API)  │    │                │
│ Decomposes  │    │ Generates     │    │ Tests & gates │    │ Commits &      │
│ features    │    │ code files    │    │ quality       │    │ pushes          │
│ into tasks  │    │ from plan     │    │ standards     │    │ to remote      │
└─────────────┘    └───────────────┘    └───────────────┘    └────────────────┘
     PLANNING       IMPLEMENTATION         QA_REVIEW            GIT_PUSH
```

### Agent Roles

| Agent | Model | Responsibility |
|-------|-------|----------------|
| **Planner** | Gemini 2.0 Flash | Decomposes a feature objective into atomic tasks with acceptance criteria, dependencies, and complexity estimates |
| **Implementer** | DeepSeek Coder (via NVIDIA API) | Generates production-ready code files from the implementation plan |
| **QA Reviewer** | Llama-3.1 8B (via NVIDIA API) | Reviews code quality, writes test cases, produces a QA report with pass/fail gating |
| **Git Controller** | GitPython | Commits verified code and pushes to the target GitHub repository |

### Pipeline Flow

1. **PLANNING** — The Planner agent receives a feature objective and produces an `ImplementationPlan` containing atomic tasks, architecture notes, and tech stack recommendations.
2. **IMPLEMENTATION** — The Implementer agent consumes the plan and generates a `CodePayload` with all required source files.
3. **QA_REVIEW** — The QA Reviewer agent writes test cases, runs them, and produces a `QAReport`. If quality gates fail, the pipeline retries (up to 3 attempts).
4. **GIT_PUSH** — The Git Controller commits all changed files and pushes to the configured remote repository.

All data flowing between agents is validated with **Pydantic v2** schemas for strict type safety.

---

## Setup

### Prerequisites

- Python 3.12+
- A GitHub repository to push code to
- API keys for NVIDIA and Google Gemini

### Installation

```bash
git clone https://github.com/CJRCODER/CJR-Swarm-Orchestrator.git
cd CJR-Swarm-Orchestrator
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or export the following variables:

```bash
export NVIDIA_API_KEY="nvapi-..."        # NVIDIA API key for DeepSeek model access
export GEMINI_API_KEY="AI..."            # Google Gemini API key
export GH_PAT="ghp_..."                 # GitHub Personal Access Token with repo scope
```

For GitHub Actions, add these as repository secrets:
- `NVIDIA_API_KEY`
- `GEMINI_API_KEY`
- `GH_PAT`

---

## Usage

### Run Locally

```bash
python swarm_agent.py
```

### Run via GitHub Actions

The swarm agent runs automatically every 2 hours via the configured GitHub Actions workflow. You can also trigger it manually from the **Actions** tab using `workflow_dispatch`.

---

## Project Structure

```
CJR-Swarm-Orchestrator/
├── .github/
│   └── workflows/
│       └── agent_loop.yml      # CI/CD workflow
├── schemas.py                  # Pydantic v2 data models
├── agents.py                   # Planner, Implementer, QA agent classes
├── git_ops.py                  # Git subprocess wrappers
├── swarm_agent.py              # Main orchestration entry point
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
