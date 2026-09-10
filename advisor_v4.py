"""Thin conversational advisor. The model understands the student; Postgres owns BCIT facts."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from database import get_connection

MODEL = os.getenv("ASTERIS_AI_MODEL", "gpt-5.6-sol")
REASONING_EFFORT = os.getenv("ASTERIS_AI_REASONING_EFFORT", "medium")
MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 12

COURSE_ID_RE = re.compile(r"\b([A-Z]{3,5})[\s\-:_]*(\d{4})\b", re.IGNORECASE)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _rows(sql: str, params: tuple = ()) -> list[tuple]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()


def _one(sql: str, params: tuple = ()) -> tuple | None:
    rows = _rows(sql, params)
    return rows[0] if rows else None


def _table_exists(name: str) -> bool:
    row = _one(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (name,),
    )
    return row is not None


def _columns(table: str) -> set[str]:
    rows = _rows(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {row[0] for row in rows}


def _pick(available: set[str], candidates: list[str]) -> list[str]:
    return [name for name in candidates if name in available]


def search_programs(
    query: str = "",
    international_only: bool = False,
    credential: str | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    cols = _columns("programs")
    select_cols = _pick(
        cols,
        [
            "program_id",
            "program_name",
            "credential",
            "school",
            "campus",
            "study_mode",
            "delivery_method",
            "status",
            "source_url",
            "program_overview",
        ],
    )
    if not select_cols:
        return {"error": "programs table is missing expected columns"}

    where = ["status = 'Active'"] if "status" in cols else ["TRUE"]
    params: list[Any] = []

    tokens = [tok for tok in re.findall(r"[a-z0-9]+", (query or "").lower()) if tok not in {
        "the", "a", "an", "i", "want", "to", "get", "into", "im", "i'm", "am",
        "what", "which", "are", "there", "my", "options", "any", "have", "do",
        "you", "about", "tell", "me", "please", "looking", "for", "some",
        "program", "programs", "student", "students",
    }]
    if tokens:
        like_parts = []
        searchable = _pick(cols, ["program_name", "program_overview", "school", "credential", "campus"])
        for token in tokens[:6]:
            token_clause = " OR ".join(f"{col} ILIKE %s" for col in searchable) or "FALSE"
            like_parts.append(f"({token_clause})")
            params.extend([f"%{token}%"] * max(len(searchable), 1))
        if like_parts:
            where.append("(" + " AND ".join(like_parts) + ")")

    if credential and "credential" in cols:
        where.append("credential ILIKE %s")
        params.append(f"%{credential}%")

    selected = ", ".join(select_cols)
    sql = f"""
        SELECT {selected}
        FROM programs
        WHERE {' AND '.join(where)}
        ORDER BY program_name
        LIMIT %s
    """
    params.append(max(1, min(limit, 25)))
    rows = _rows(sql, tuple(params))
    programs = [dict(zip(select_cols, row)) for row in rows]

    if international_only:
        annotated = []
        for program in programs:
            status = get_international_status(program.get("program_id", ""))
            program["international"] = status
            if str(status.get("state", "")).upper() in {
                "ACCEPTED_AVAILABLE",
                "CONDITIONAL_RESTRICTED",
                "AVAILABLE",
                "YES",
            }:
                annotated.append(program)
        if annotated:
            programs = annotated
        else:
            for program in programs:
                program["international"] = get_international_status(program.get("program_id", ""))

    return {
        "match_count": len(programs),
        "query_tokens": tokens,
        "programs": [
            {
                "program_id": item.get("program_id"),
                "program_name": item.get("program_name"),
                "credential": item.get("credential"),
                "school": item.get("school"),
                "campus": item.get("campus"),
                "study_mode": item.get("study_mode"),
                "source_url": item.get("source_url"),
                "overview": (item.get("program_overview") or "")[:280],
                "international": item.get("international"),
            }
            for item in programs
        ],
    }


def get_program(program_id: str) -> dict[str, Any]:
    if not program_id:
        return {"error": "program_id is required"}
    cols = _columns("programs")
    select_cols = _pick(
        cols,
        [
            "program_id",
            "program_name",
            "program_overview",
            "school",
            "credential",
            "study_mode",
            "campus",
            "delivery_method",
            "status",
            "source_url",
        ],
    )
    row = _one(
        f"SELECT {', '.join(select_cols)} FROM programs WHERE program_id = %s",
        (program_id.upper(),),
    )
    if row is None:
        return {"error": "program_not_found", "program_id": program_id}
    program = dict(zip(select_cols, row))
    course_rows = _rows(
        """
        SELECT pc.course_id, c.course_name, pc.level, pc.required, pc.course_type
        FROM program_courses pc
        JOIN courses c ON c.course_id = pc.course_id
        WHERE pc.program_id = %s
        ORDER BY pc.level NULLS LAST, pc.course_id
        """,
        (program_id.upper(),),
    )
    courses = [
        {
            "course_id": item[0],
            "course_name": item[1],
            "level": item[2],
            "required": item[3],
            "course_type": item[4],
        }
        for item in course_rows
    ]
    return {
        "program": program,
        "course_count": len(courses),
        "sample_courses": courses[:12],
        "note": "Full course lists are available on request; this is a summary card.",
        "international": get_international_status(program_id),
        "admission": get_admission_rules(program_id),
    }


def get_admission_rules(program_id: str) -> dict[str, Any]:
    if not program_id:
        return {"error": "program_id is required"}
    if _table_exists("academic_rule_conditions"):
        rows = _rows(
            """
            SELECT rs.rule_set_id, rs.scope, ac.condition_id, ac.condition_type,
                   ac.description, ac.parameters
            FROM academic_rule_sets rs
            JOIN academic_rule_conditions ac ON ac.rule_set_id = rs.rule_set_id
            WHERE rs.program_id = %s
            ORDER BY rs.rule_set_id, ac.condition_id
            LIMIT 40
            """,
            (program_id.upper(),),
        )
        if rows:
            return {
                "program_id": program_id.upper(),
                "source": "academic_rule_conditions",
                "conditions": [
                    {
                        "rule_set_id": row[0],
                        "scope": row[1],
                        "condition_id": row[2],
                        "type": row[3],
                        "description": row[4],
                        "parameters": row[5],
                    }
                    for row in rows
                ],
            }
    if _table_exists("program_requirements"):
        req_cols = _columns("program_requirements")
        select_cols = _pick(
            req_cols,
            ["requirement_type", "notes", "description", "level", "course_id", "required"],
        )
        if select_cols:
            rows = _rows(
                f"""
                SELECT {', '.join(select_cols)}
                FROM program_requirements
                WHERE program_id = %s
                LIMIT 40
                """,
                (program_id.upper(),),
            )
            return {
                "program_id": program_id.upper(),
                "source": "program_requirements",
                "conditions": [dict(zip(select_cols, row)) for row in rows],
            }
    return {
        "program_id": program_id.upper(),
        "state": "UNKNOWN_NOT_PUBLISHED",
        "message": "No structured admission rules were found for this program in the local schema snapshot.",
    }


def get_international_status(program_id: str) -> dict[str, Any]:
    if not program_id:
        return {"error": "program_id is required"}
    for table in (
        "program_international_status",
        "international_program_status",
        "program_delivery_facts",
        "international_eligibility",
    ):
        if not _table_exists(table):
            continue
        cols = _columns(table)
        if "program_id" not in cols:
            continue
        select_cols = _pick(
            cols,
            [
                "program_id",
                "status",
                "state",
                "international_status",
                "eligibility",
                "notes",
                "restriction",
                "source_url",
            ],
        )
        row = _one(
            f"SELECT {', '.join(select_cols)} FROM {table} WHERE program_id = %s LIMIT 1",
            (program_id.upper(),),
        )
        if row:
            data = dict(zip(select_cols, row))
            data["table"] = table
            return data
    prog_cols = _columns("programs")
    extra = _pick(prog_cols, ["international_status", "international_eligible", "notes"])
    if extra:
        row = _one(
            f"SELECT {', '.join(extra)} FROM programs WHERE program_id = %s",
            (program_id.upper(),),
        )
        if row and any(row):
            return dict(zip(extra, row))
    return {
        "program_id": program_id.upper(),
        "state": "UNKNOWN_NOT_PUBLISHED",
        "message": "No published international decision is stored for this program.",
    }


def list_campuses() -> dict[str, Any]:
    for table in ("campuses", "campus", "campus_directory", "institutional_campuses"):
        if not _table_exists(table):
            continue
        cols = _columns(table)
        select_cols = _pick(
            cols,
            ["campus_id", "name", "campus_name", "address", "city", "phone", "description", "source_url"],
        )
        if not select_cols:
            continue
        rows = _rows(f"SELECT {', '.join(select_cols)} FROM {table} ORDER BY 1")
        campuses = [dict(zip(select_cols, row)) for row in rows]
        return {"count": len(campuses), "campuses": campuses, "source_table": table}

    if "campus" in _columns("programs"):
        rows = _rows(
            """
            SELECT campus, COUNT(*)
            FROM programs
            WHERE status = 'Active' AND campus IS NOT NULL AND campus <> ''
            GROUP BY campus
            ORDER BY campus
            """
        )
        campuses = [{"name": row[0], "active_program_count": row[1]} for row in rows]
        return {
            "count": len(campuses),
            "campuses": campuses,
            "source_table": "programs.campus",
            "note": "Names come from active program records because a campus directory table was not found.",
        }
    return {"count": 0, "campuses": [], "state": "UNKNOWN_NOT_PUBLISHED"}


def search_courses(query: str) -> dict[str, Any]:
    if not query.strip():
        return {"courses": []}
    rows = _rows(
        """
        SELECT course_id, course_name, credits, status
        FROM courses
        WHERE status = 'Active'
          AND (course_id ILIKE %s OR course_name ILIKE %s)
        ORDER BY
            CASE WHEN UPPER(course_id) = UPPER(%s) THEN 0
                 WHEN LOWER(course_name) = LOWER(%s) THEN 1
                 ELSE 2 END,
            course_id
        LIMIT 12
        """,
        (f"%{query}%", f"%{query}%", query, query),
    )
    return {
        "courses": [
            {"course_id": row[0], "course_name": row[1], "credits": row[2], "status": row[3]}
            for row in rows
        ]
    }


def get_course(course_id: str) -> dict[str, Any]:
    if not course_id:
        return {"error": "course_id is required"}
    row = _one(
        """
        SELECT course_id, course_name, credits, course_overview, status, source_url, notes
        FROM courses
        WHERE course_id = %s
        """,
        (course_id.upper(),),
    )
    if row is None:
        return {"error": "course_not_found", "course_id": course_id}
    prereq_rows = _rows(
        """
        SELECT pg.group_type, pc.prerequisite_course_id, pc.minimum_grade,
               pc.required_program_id, pc.condition_type, pc.notes
        FROM prerequisite_groups pg
        JOIN prerequisite_conditions pc ON pc.prerequisite_group_id = pg.prerequisite_group_id
        WHERE pg.course_id = %s
        ORDER BY pg.prerequisite_group_id, pc.prerequisite_condition_id
        """,
        (course_id.upper(),),
    )
    return {
        "course": {
            "course_id": row[0],
            "course_name": row[1],
            "credits": row[2],
            "overview": row[3],
            "status": row[4],
            "source_url": row[5],
            "notes": row[6],
        },
        "prerequisites": [
            {
                "group_type": item[0],
                "course_id": item[1],
                "minimum_grade": item[2],
                "required_program_id": item[3],
                "condition_type": item[4],
                "notes": item[5],
            }
            for item in prereq_rows
        ],
    }


def evaluate_eligibility(program_id: str | None, student_profile: dict[str, Any] | None) -> dict[str, Any]:
    if not program_id:
        return {"error": "program_id is required"}
    profile = student_profile or {}
    admission = get_admission_rules(program_id)
    international = get_international_status(program_id)
    facts = []
    for condition in admission.get("conditions") or []:
        description = str(condition.get("description") or condition.get("notes") or condition.get("type") or "")
        facts.append({"condition": condition, "description": description})
    return {
        "program_id": program_id.upper(),
        "student_profile": profile,
        "admission": admission,
        "international": international,
        "note": (
            "Compare the student profile only against published stored conditions. "
            "Mark MET / UNMET / UNKNOWN. Do not invent missing requirements. "
            "Human-confirmation items stay human-confirmation items."
        ),
        "condition_summaries": facts[:30],
    }


TOOLS = [
    {
        "type": "function",
        "name": "search_programs",
        "description": (
            "Search active programs by meaning, not exact title only. "
            "Use for finance, computing, nursing, international options, and similar discovery."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "international_only": {"type": "boolean"},
                "credential": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_program",
        "description": "Get one program summary, international state, and admission evidence.",
        "parameters": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_admission_rules",
        "description": "Get stored admission or program-requirement evidence for one program.",
        "parameters": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_international_status",
        "description": "Get the stored international availability state for one program.",
        "parameters": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_campuses",
        "description": "List BCIT campuses from the stored directory or program records.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "search_courses",
        "description": "Find active courses by code or name.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_course",
        "description": "Get one course and its stored prerequisites.",
        "parameters": {
            "type": "object",
            "properties": {"course_id": {"type": "string"}},
            "required": ["course_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "evaluate_eligibility",
        "description": "Compare a student's stated credentials with stored program rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "program_id": {"type": "string"},
                "student_profile": {"type": "object"},
            },
            "required": ["program_id"],
            "additionalProperties": False,
        },
    },
]


TOOL_IMPL = {
    "search_programs": lambda args, state: search_programs(
        query=str(args.get("query") or ""),
        international_only=bool(args.get("international_only")),
        credential=args.get("credential"),
        limit=int(args.get("limit") or 12),
    ),
    "get_program": lambda args, state: get_program(str(args.get("program_id") or "")),
    "get_admission_rules": lambda args, state: get_admission_rules(str(args.get("program_id") or "")),
    "get_international_status": lambda args, state: get_international_status(str(args.get("program_id") or "")),
    "list_campuses": lambda args, state: list_campuses(),
    "search_courses": lambda args, state: search_courses(str(args.get("query") or "")),
    "get_course": lambda args, state: get_course(str(args.get("course_id") or "")),
    "evaluate_eligibility": lambda args, state: evaluate_eligibility(
        args.get("program_id"),
        args.get("student_profile") if isinstance(args.get("student_profile"), dict) else state.get("student"),
    ),
}


INSTRUCTIONS = """
You are Asteris, a BCIT academic advisor.

Understand the student's current message first. Use conversation memory only for follow-ups like "this one", "which one?", "that program", or added grades.

PostgreSQL is the only source of BCIT facts. Use tools. Never invent programs, campuses, grades, admission rules, or international status.

How to answer:
- Sound like a competent human advisor. Warm, direct, specific.
- Do not say "stored campus directory", "verified packet", "tool result", or "academic facts checked".
- Broad questions: list a short set of relevant programs, then ask which to open.
- Exact program questions: open that program and answer the asked facet.
- International questions: use stored four-state evidence. UNKNOWN_NOT_PUBLISHED means unpublished, not unavailable.
- If a lookup fails, say you could not retrieve it. That is different from "BCIT does not offer it".
- If the student gives grades or credentials, keep those as student facts and compare them with stored rules.
- If a requirement needs a person or committee, say that plainly.
- Prefer 1-3 short paragraphs or a tight list. Do not dump raw tables.
- BCIT offers and requires. Asteris only advises.
"""


def _normalize_state(state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(state or {})
    state.setdefault("student", {})
    state.setdefault("current_program_id", None)
    state.setdefault("last_listed_program_ids", [])
    state.setdefault("last_listed_course_ids", [])
    state.setdefault("last_listed_campuses", [])
    return state


def _update_state(state: dict[str, Any], tool_name: str, result: Any) -> None:
    if not isinstance(result, dict):
        return
    if tool_name == "search_programs":
        ids = [item.get("program_id") for item in result.get("programs") or [] if item.get("program_id")]
        if ids:
            state["last_listed_program_ids"] = ids
    elif tool_name == "get_program" and result.get("program", {}).get("program_id"):
        state["current_program_id"] = result["program"]["program_id"]
        state["last_listed_program_ids"] = [result["program"]["program_id"]]
    elif tool_name == "list_campuses":
        names = []
        for campus in result.get("campuses") or []:
            names.append(campus.get("name") or campus.get("campus_name"))
        state["last_listed_campuses"] = [name for name in names if name]
    elif tool_name == "search_courses":
        ids = [item.get("course_id") for item in result.get("courses") or [] if item.get("course_id")]
        if ids:
            state["last_listed_course_ids"] = ids


def _extract_student_hints(question: str, state: dict[str, Any]) -> None:
    text = question.lower()
    student = state.setdefault("student", {})
    if "international" in text:
        student["international"] = True
    if re.search(r"english\s*12", text):
        student["english_12"] = question
    if "work experience" in text or "no work" in text:
        student["work_experience_note"] = question
    if "diploma" in text:
        student["diploma_note"] = question
    match = re.search(r"(\d{2})\s*%", text)
    if match:
        student["last_grade_percent"] = int(match.group(1))


def answer_student_question_v4(
    question: str,
    conversation: list[dict[str, str]] | None = None,
    advisor_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty")

    state = _normalize_state(advisor_state)
    _extract_student_hints(question, state)

    history_lines = []
    for message in (conversation or [])[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role}: {content}")

    user_input = (
        f"Current student message:\n{question}\n\n"
        f"Structured memory:\n{json.dumps(state, default=str)}\n\n"
        f"Recent conversation:\n" + ("\n".join(history_lines) if history_lines else "(none)")
    )

    client = _client()
    request = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "tools": TOOLS,
    }
    if REASONING_EFFORT:
        request["reasoning"] = {"effort": REASONING_EFFORT}

    response = client.responses.create(**request)
    tools_used: list[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        calls = [item for item in response.output if getattr(item, "type", "") == "function_call"]
        if not calls:
            return {
                "answer": (response.output_text or "").strip(),
                "tools_used": tools_used,
                "advisor_state": state,
                "version": "v4",
            }

        outputs = []
        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
                result = TOOL_IMPL[call.name](arguments, state)
            except Exception as error:  # noqa: BLE001
                result = {"error": f"{type(error).__name__}: {error}"}
            tools_used.append(call.name)
            _update_state(state, call.name, result)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                }
            )

        follow = {
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "previous_response_id": response.id,
            "input": outputs,
            "tools": TOOLS,
        }
        if REASONING_EFFORT:
            follow["reasoning"] = {"effort": REASONING_EFFORT}
        response = client.responses.create(**follow)

    return {
        "answer": "I started the lookup but could not finish it cleanly. Please ask that again in one specific question.",
        "tools_used": tools_used,
        "advisor_state": state,
        "version": "v4",
        "error": "tool_round_limit_reached",
    }
