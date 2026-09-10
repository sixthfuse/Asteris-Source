"""Natural-language advisor backed by small, verified Asteris tools."""

import json
import os
import re
from enum import Enum
from typing import Any

from openai import OpenAI

from advisor import find_courses, get_course_details
from advisor_engine import run_advisor_engine
from database import get_connection
from eligibility import check_course_eligibility
from program_requirements import check_program_progress


MODEL = os.getenv("ASTERIS_AI_MODEL", "gpt-5.6-luna")
INSTITUTION_NAME = os.getenv("ASTERIS_INSTITUTION_NAME", "the institution")
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARACTERS = 4_000


class AdvisorIntent(str, Enum):
    COURSE_INFO = "COURSE_INFO"
    PROGRAM_INFO = "PROGRAM_INFO"
    PREREQUISITES = "PREREQUISITES"
    ELIGIBILITY_NEXT_COURSES = "ELIGIBILITY_NEXT_COURSES"
    PROGRAM_PROGRESS = "PROGRAM_PROGRESS"
    PROGRAM_CATEGORY = "PROGRAM_CATEGORY"
    ELECTIVES = "ELECTIVES"
    PROGRESSION = "PROGRESSION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    FALLBACK = "FALLBACK"


COMPLETION_CUES = re.compile(
    r"\b(?:completed?|finished|passed|took|taken|done|earned|"
    r"complet[ée]|termin[ée]|aprob[ée]|curs[ée]|finalic[ée])\b",
    re.IGNORECASE,
)
COURSE_ID_PATTERN = re.compile(r"\b([A-Z]{3,5})[\s\-:_]*(\d{4})\b", re.IGNORECASE)
LEVEL_NUMBER_PATTERN = re.compile(
    r"\b(?:level|nivel)\s*(one|two|three|four|five|six|seven|eight|"
    r"uno|dos|tres|cuatro|cinco|seis|siete|ocho|\d+|&)\b",
    re.IGNORECASE,
)

GLOBAL_CATALOG_PATTERN = re.compile(
    r"\b(?:all|every|full|complete)\s+(?:active\s+)?(?:programs?|credentials?)\b|"
    r"\bhow\s+many\s+(?:active\s+)?programs?\b|"
    r"\b(?:what|which)\s+(?:active\s+)?(?:programs?|credentials?)\s+"
    r"(?:do\s+you\s+(?:offer|have)|are\s+(?:available|offered))\b|"
    r"\b(?:list|show|give\s+me(?:\s+a\s+list\s+of)?)\s+"
    r"(?:all\s+)?(?:active\s+)?(?:programs?|credentials?)\b|"
    r"\bcu[aá]nt(?:os|as)\s+(?:programas?|carreras?)\s+(?:ofrec(?:e|en)|hay)\b|"
    r"\b(?:qu[eé]|cu[aá]les)\s+(?:programas?|carreras?|credenciales?)\s+"
    r"(?:ofrec(?:e|en)|tien(?:e|en)|hay|est[aá]n\s+disponibles)\b|"
    r"\b(?:lista(?:do)?\s+de|listar|mu[eé]strame|dame(?:\s+una\s+lista\s+de)?)\s+"
    r"(?:todos\s+los\s+|todas\s+las\s+)?(?:programas?|carreras?|credenciales?)\b",
    re.IGNORECASE,
)

SPANISH_MARKERS = re.compile(
    r"[¿¡áéíóúñü]|\b(?:cu[aá]ntos?|cu[aá]les?|qu[eé]|programas?|carreras?|"
    r"ofrecen?|lista(?:do)?|cursos?|requisitos?|nivel|puedo|necesito)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE_PATTERNS = (
    re.compile(r"\bshakira\b", re.IGNORECASE),
    re.compile(r"\b(?:world cup|copa (?:del )?mundo|fifa)\b", re.IGNORECASE),
    re.compile(r"\b(?:weather|forecast|temperatur[ae]|clima|pron[oó]stico)\b", re.IGNORECASE),
    re.compile(r"\b(?:prime minister|president|election|politics|pol[ií]tica|elecciones?)\b", re.IGNORECASE),
    re.compile(r"\b(?:celebrity|celebrities|movie|movies|music|singer|actor|actress|"
               r"pel[ií]cula|m[uú]sica|cantante|actor|actriz)\b", re.IGNORECASE),
)

ACADEMIC_SCOPE_PATTERN = re.compile(
    r"\b(?:bcit|institution|institutional|academic|student|career|careers|job|jobs|"
    r"program|programs|programa|programas|carrera|carreras|course|courses|curso|cursos|"
    r"admission|admissions|admisi[oó]n|prerequisite|prerequisites|requisito|requisitos|"
    r"credential|credentials|degree|diploma|microcredential|engineering|college|school|"
    r"campus|tuition|enrol|enroll|apply|application|graduate|graduation|elective|"
    r"level|nivel|credit|credits|transfer|schedule)\b",
    re.IGNORECASE,
)


def current_message_language(question: str) -> str:
    """Detect the response language from this turn, never from conversation history."""
    return "Spanish" if SPANISH_MARKERS.search(question) else "English"


def is_out_of_scope_question(question: str) -> bool:
    """Reject clearly unrelated general knowledge without blocking academic questions."""
    return not ACADEMIC_SCOPE_PATTERN.search(question) and any(
        pattern.search(question) for pattern in OUT_OF_SCOPE_PATTERNS
    )


def out_of_scope_response(question: str) -> str:
    if current_message_language(question) == "Spanish":
        return (
            "Estoy enfocado en la orientación académica e institucional de BCIT. "
            "Puedo ayudarte con programas, cursos, admisiones, requisitos previos y planificación académica."
        )
    return (
        "I'm focused on BCIT academic and institutional advising. I can help with "
        "programs, courses, admissions, prerequisites, and academic planning."
    )

PROGRAM_ID_REQUEST_PATTERN = re.compile(
    r"\bprogram\s+(?:ids?|codes?)\b|(?<!course\s)(?<!course-)"
    r"\b(?:ids?|codes?)\b(?=[^.!?]{0,50}\bprograms?\b)",
    re.IGNORECASE,
)
PROGRAM_LINK_REQUEST_PATTERN = re.compile(
    r"\b(?:links?|urls?|web(?:site)?\s+links?)\b", re.IGNORECASE
)


def is_global_catalog_question(question: str) -> bool:
    """Return true when this turn explicitly asks about the whole catalog."""
    text = re.sub(r"\s+", " ", question.strip())
    return bool(
        GLOBAL_CATALOG_PATTERN.search(text)
        or ((PROGRAM_ID_REQUEST_PATTERN.search(text) or PROGRAM_LINK_REQUEST_PATTERN.search(text))
            and re.search(r"\b(?:all|the|three|3)\s+(?:active\s+)?programs?\b", text, re.I))
    )


def explicitly_requests_program_ids(question: str) -> bool:
    return bool(PROGRAM_ID_REQUEST_PATTERN.search(question))


def explicitly_requests_program_links(question: str) -> bool:
    return bool(PROGRAM_LINK_REQUEST_PATTERN.search(question))


TOOLS = [
    {
        "type": "function",
        "name": "search_courses",
        "description": "Find active courses by course code or course-name text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_course_details",
        "description": "Get verified details and prerequisites for one course code.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
            },
            "required": ["course_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_programs",
        "description": "Resolve a friendly program name to an active Asteris program record.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_program_details",
        "description": "Get verified program-level information for one resolved program.",
        "parameters": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_program_level_courses",
        "description": (
            "Get the verified course list for a requested level of the already "
            "resolved program or shared academic pathway."
        ),
        "parameters": {
            "type": "object",
            "properties": {"level": {"type": "integer", "minimum": 1}},
            "required": ["level"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function", "name": "get_program_electives",
        "description": "List verified elective choice groups for the resolved program; work terms are excluded.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_program_progression_requirements",
        "description": "Get verified academic and practical-work progression requirements for the resolved program.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "check_course_eligibility",
        "description": (
            "Check whether the student may take one course, including prerequisites "
            "and program-level progression when a program is known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
            },
            "required": ["course_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_student_program_advice",
        "description": (
            "Calculate program progress and the courses the student can take next."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_student_level_readiness",
        "description": (
            "List the exact unfinished required courses before a requested level, "
            "using persisted completed-course state."
        ),
        "parameters": {
            "type": "object",
            "properties": {"target_level": {"type": "integer", "minimum": 2}},
            "required": ["target_level"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


TOOLS_BY_INTENT = {
    AdvisorIntent.COURSE_INFO: {"search_courses", "get_course_details"},
    AdvisorIntent.PROGRAM_INFO: {
        "search_programs", "get_program_details", "get_program_level_courses"
    },
    AdvisorIntent.PREREQUISITES: {"search_courses", "get_course_details"},
    AdvisorIntent.ELIGIBILITY_NEXT_COURSES: {
        "search_courses", "check_course_eligibility", "get_student_program_advice"
    },
    AdvisorIntent.PROGRAM_PROGRESS: {
        "get_student_program_advice", "get_student_level_readiness"
    },
    AdvisorIntent.PROGRAM_CATEGORY: {"search_programs"},
    AdvisorIntent.ELECTIVES: {"search_programs", "get_program_electives"},
    AdvisorIntent.PROGRESSION: {"search_programs", "get_program_progression_requirements"},
    AdvisorIntent.FALLBACK: {
        "search_courses", "get_course_details", "search_programs", "get_program_details"
    },
}


def classify_intent(question: str) -> AdvisorIntent:
    """Route common student language before exposing any academic tools."""

    text = re.sub(r"\s+", " ", question.strip().lower())
    if is_out_of_scope_question(question):
        return AdvisorIntent.OUT_OF_SCOPE
    if is_global_catalog_question(question):
        return AdvisorIntent.PROGRAM_CATEGORY
    if any(phrase in text for phrase in (
        "more information", "more details", "program details", "course names",
        "course ids", "course codes", "courses in", "courses for",
    )) and ("program" in text or "microcredential" in text or "micro-credential" in text):
        return AdvisorIntent.PROGRAM_INFO
    if any(phrase in text for phrase in (
        "what do i still need", "what am i missing", "what do i need for level",
        "what do i need to finish", "what remains to finish", "finish level",
        "requirements remain", "requirements are left", "requirements are remaining",
        "qué me falta", "que me falta", "qué necesito",
        "que necesito", "para el nivel", "por completar",
    )):
        return AdvisorIntent.PROGRAM_PROGRESS
    if any(word in text for word in ("elective", "choice group", "choose from")):
        return AdvisorIntent.ELECTIVES
    if any(phrase in text for phrase in (
        "practical work", "work experience", "continue to level", "progression requirement",
        "progress to level", "move to level", "advance to level",
        "experiencia laboral", "trabajo práctico", "trabajo practico",
        "avanzar al nivel", "pasar al nivel",
    )):
        return AdvisorIntent.PROGRESSION
    if any(phrase in text for phrase in (
        "how much of my program", "program progress", "progress toward",
        "requirements remain", "left to graduate", "am i finished",
        "what percentage of the program", "percentage of my program",
        "percentage of the whole", "percentage of the program",
        "porcentaje del programa", "cuánto he completado", "cuanto he completado",
    )):
        return AdvisorIntent.PROGRAM_PROGRESS
    if any(phrase in text for phrase in (
        "can i take", "am i eligible", "eligible for", "take next",
        "courses next", "what should i take", "what can i take",
        "qué puedo tomar", "que puedo tomar", "qué cursos siguen", "que cursos siguen",
    )):
        return AdvisorIntent.ELIGIBILITY_NEXT_COURSES
    if any(phrase in text for phrase in (
        "prerequisite", "prerequisites", "what do i need before",
        "required before", "requirements for the course",
    )):
        return AdvisorIntent.PREREQUISITES
    if any(phrase in text for phrase in (
        "microcredential", "micro-credential", "what credentials", "which programs",
        "what programs", "programs do you have", "programs are available",
        "qué programas", "que programas", "cuáles programas", "cuales programas",
        "qué carreras", "que carreras", "cuáles carreras", "cuales carreras",
    )):
        return AdvisorIntent.PROGRAM_CATEGORY
    if any(word in text for word in ("program", "programa", "carrera", "diploma", "degree")):
        return AdvisorIntent.PROGRAM_INFO
    if re.fullmatch(r"[a-z]{3,5}[ -]?\d{4}", text) or len(text.split()) <= 8:
        return AdvisorIntent.COURSE_INFO
    return AdvisorIntent.FALLBACK


def find_programs(search_text: str) -> list[dict[str, Any]]:
    """Resolve natural names while keeping database IDs server-side."""

    ignored = {"the", "a", "an", "all", "active", "what", "which", "program", "programs", "credential", "credentials", "list", "how", "many", "more", "information", "about", "tell", "me", "give", "on", "do", "you", "have", "offer", "any", "show", "available", "are", "qué", "que", "cuál", "cual", "cuáles", "cuales", "cuánto", "cuanto", "cuántos", "cuantos", "cuánta", "cuanta", "cuántas", "cuantas", "programa", "programas", "carrera", "carreras", "credencial", "credenciales", "lista", "listado", "todos", "todas", "los", "las", "ofrece", "ofrecen", "tiene", "tienen", "hay", "disponibles", "en", "de", "una", "un", "dame", "muestra", "muéstrame"}
    words = [word for word in re.findall(r"[a-z0-9]+", search_text.lower()) if word not in ignored]
    category_aliases = {
        "microcredential": "microcredential", "microcredentials": "microcredential", "micro": "microcredential",
        "diploma": "diploma", "degree": "degree", "bachelor": "bachelor",
    }
    category = next((value for key, value in category_aliases.items() if key in words), None)
    if is_global_catalog_question(search_text) or not words:
        pattern = "%"
    else:
        pattern = "%" + "%".join(words) + "%"
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if category:
                cursor.execute(
                    """
                    SELECT program_id, program_name, credential, study_mode, campus, source_url
                    FROM programs
                    WHERE status = 'Active' AND credential ILIKE %s
                    ORDER BY program_name LIMIT 10
                    """, (f"%{category}%",),
                )
            else:
                cursor.execute(
                    """
                    SELECT program_id, program_name, credential, study_mode, campus, source_url
                    FROM programs
                    WHERE status = 'Active' AND program_name ILIKE %s
                    ORDER BY CASE WHEN LOWER(program_name) = LOWER(%s) THEN 0 ELSE 1 END,
                             program_name LIMIT 10
                    """, (pattern, search_text.strip()),
                )
            rows = cursor.fetchall()
    return [
        {"program_id": row[0], "program_name": row[1], "credential": row[2],
         "study_mode": row[3], "campus": row[4], "source_url": row[5]}
        for row in rows
    ]


def get_program_details(program_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT program_id, program_name, program_overview, school,
                       credential, study_mode, campus, delivery_method, status, source_url
                FROM programs WHERE program_id = %s
                """,
                (program_id.upper(),),
            )
            row = cursor.fetchone()
            cursor.execute(
                """
                SELECT pc.course_id, c.course_name, c.credits, c.source_url,
                       pc.level, pc.term, pc.course_type, pc.required, pc.notes
                FROM program_courses pc
                JOIN courses c ON c.course_id = pc.course_id
                WHERE pc.program_id = %s
                ORDER BY pc.level NULLS LAST, pc.course_id
                """,
                (program_id.upper(),),
            )
            course_rows = cursor.fetchall()
    if row is None:
        return None
    return {
        "program_id": row[0], "program_name": row[1], "program_overview": row[2],
        "school": row[3], "credential": row[4], "study_mode": row[5],
        "campus": row[6], "delivery_method": row[7], "status": row[8],
        "source_url": row[9],
        "courses": [
            {"course_id": item[0], "course_name": item[1], "credits": item[2],
             "source_url": item[3], "level": item[4], "term": item[5],
             "course_type": item[6], "required": item[7], "notes": item[8]}
            for item in course_rows
        ],
    }


def get_program_electives(
    program_id: str, level: int | None = None
) -> list[dict[str, Any]]:
    """Return actual choice groups, never optional work terms."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT pr.choice_group, pr.choice_required, pr.level, pr.requirement_type,
                       pr.notes, pr.course_id, c.course_name
                FROM program_requirements pr
                JOIN courses c ON c.course_id = pr.course_id
                WHERE pr.program_id = %s AND pr.choice_group IS NOT NULL
                  AND LOWER(pr.requirement_type) NOT LIKE '%%work term%%'
                  AND (%s::integer IS NULL OR pr.level = %s::integer)
                ORDER BY pr.level, pr.choice_group, pr.course_id
                """, (program_id.upper(), level, level),
            )
            rows = cursor.fetchall()
    groups: dict[str, dict[str, Any]] = {}
    for group_id, required, level, kind, notes, course_id, course_name in rows:
        group = groups.setdefault(group_id, {
            "level": level, "type": kind, "choose": required, "description": notes, "courses": []
        })
        group["courses"].append({"course_id": course_id, "course_name": course_name})
    return list(groups.values())


def get_program_progression_requirements(program_id: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT from_level, to_level, requirement_type, requirement_value,
                       description, source_url
                FROM progression_requirements WHERE program_id = %s
                ORDER BY from_level, progression_requirement_id
                """, (program_id.upper(),),
            )
            rows = cursor.fetchall()
    return [{"from_level": r[0], "to_level": r[1], "type": r[2], "value": r[3],
             "description": r[4], "source_url": r[5]} for r in rows]


def get_program_level_courses(program_id: str, level: int) -> list[dict[str, Any]]:
    """Return enriched, student-facing course records for one program level."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT pc.course_id, c.course_name, c.credits, c.source_url,
                       pc.level, pc.term, pc.course_type, pc.required, pc.notes
                FROM program_courses pc
                JOIN courses c ON c.course_id = pc.course_id
                WHERE pc.program_id = %s AND pc.level = %s
                ORDER BY pc.course_id
                """,
                (program_id.upper(), level),
            )
            rows = cursor.fetchall()
    return [
        {"course_id": row[0], "course_name": row[1], "credits": row[2],
         "source_url": row[3], "level": row[4], "term": row[5],
         "course_type": row[6], "required": row[7], "notes": row[8]}
        for row in rows
    ]


def resolve_academic_context(
    question: str,
    conversation: list[dict[str, str]] | None,
    selected_program_id: str | None,
) -> dict[str, Any]:
    """Resolve a friendly program mention from this turn or recent conversation."""

    global_catalog = is_global_catalog_question(question)
    history = " ".join(message.get("content", "") for message in (conversation or []))
    full_text = re.sub(r"\s+", " ", f"{history} {question}".lower())
    question_text = re.sub(r"\s+", " ", question.lower())

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT program_id, program_name, credential
                FROM programs
                WHERE status = 'Active'
                ORDER BY program_name
                """
            )
            programs = cursor.fetchall()

    stop = {"program", "engineering", "of", "the", "and", "degree", "diploma", "bachelor"}

    def score(row, text):
        tokens = {t for t in re.findall(r"[a-z0-9]+", row[1].lower()) if t not in stop}
        return sum(token in text for token in tokens)

    explicit = [(score(row, question_text), row) for row in programs]
    explicit = [item for item in explicit if item[0] > 0]
    historical = [(score(row, full_text), row) for row in programs]
    historical = [item for item in historical if item[0] > 0]
    candidate = None if global_catalog else max(
        explicit or historical, default=(0, None), key=lambda item: item[0]
    )[1]

    wanted = None
    for term, credential in (("bachelor", "bachelor"), ("degree", "bachelor"),
                             ("diploma", "diploma"), ("microcredential", "microcredential"),
                             ("micro-credential", "microcredential")):
        if term in question_text:
            wanted = credential
            break
    if wanted and any(subject in full_text for subject in ("civil", "circular", "economy")):
        matches = [row for row in programs if wanted in (row[2] or "").lower()]
        if matches:
            candidate = max(matches, key=lambda row: score(row, full_text))

    # A credential-only reference can identify a program when the catalog has a
    # single active match. This makes natural singular references (and their
    # follow-ups) resolve to the real internal relationship key without exposing it.
    credential_term = next(
        (credential for term, credential in (
            ("microcredential", "microcredential"),
            ("micro-credential", "microcredential"),
            ("diploma", "diploma"),
            ("bachelor", "bachelor"),
        ) if term in full_text),
        None,
    )
    if not candidate and credential_term:
        credential_matches = [
            row for row in programs if credential_term in (row[2] or "").lower()
        ]
        if len(credential_matches) == 1:
            candidate = credential_matches[0]

    if global_catalog:
        candidate = None
    resolved = None if global_catalog else (candidate[0] if candidate else selected_program_id)
    program_name = candidate[1] if candidate else None
    if selected_program_id and not candidate and not global_catalog:
        selected = next((row for row in programs if row[0] == selected_program_id), None)
        if selected:
            program_name = selected[1]
    pathway = {"name": "Civil Engineering"} if "civil engineering" in full_text else None
    if pathway and not resolved:
        bachelor = next((row for row in programs if "civil engineering" in row[1].lower()
                         and "bachelor" in (row[2] or "").lower()), None)
        if bachelor:
            resolved, program_name = bachelor[0], bachelor[1]
    # Academic references are resolved newest-first. The current turn overrides
    # history, while elliptical follow-ups inherit the most recent structured value.
    texts = [question] + [
        message.get("content", "") for message in reversed(conversation or [])
        if message.get("role") in {"user", "assistant"}
    ]
    active_level = next(
        (requested_target_level(text) for text in texts if requested_target_level(text)),
        None,
    )
    topic = next(
        ("electives" for text in texts
         if re.search(r"\b(?:electives?|choice groups?|choose from)\b", text, re.I)),
        None,
    )
    course = next(
        (normalize_course_id(match.group(0)) for text in texts
         if (match := COURSE_ID_PATTERN.search(text))),
        None,
    )
    return {
        "program_id": resolved, "program_name": program_name, "pathway": pathway,
        "level": active_level, "requirement_type": topic, "course_id": course,
        "global_catalog": global_catalog,
    }


def resolve_civil_engineering_context(question, conversation, selected_program_id):
    """Compatibility wrapper for callers from earlier phases."""
    return resolve_academic_context(question, conversation, selected_program_id)


def infer_completed_levels(
    question: str, program_id: str | None, completed_courses: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Turn an explicit 'finished all level N courses' statement into engine input."""

    if not program_id:
        return completed_courses, []
    matches = re.findall(
        r"(?:finished|completed|passed)\s+all\s+(?:my\s+)?level\s+"
        r"(one|two|three|four|five|six|seven|eight|\d+)(?:\s+courses)?",
        question.lower(),
    )
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8,
    }
    stated_levels = sorted({number_words.get(value, int(value) if value.isdigit() else 0) for value in matches})
    stated_levels = [level for level in stated_levels if level > 0]
    if not stated_levels:
        return completed_courses, []

    highest_level = max(stated_levels)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT course_id
                FROM program_courses
                WHERE program_id = %s AND level <= %s AND required = TRUE
                ORDER BY level, course_id
                """,
                (program_id.upper(), highest_level),
            )
            inferred_ids = [row[0] for row in cursor.fetchall()]

    merged = {normalize_course_id(item["course_id"]): dict(item) for item in completed_courses}
    for course_id in inferred_ids:
        merged.setdefault(course_id, {"course_id": course_id, "grade": None})
    return list(merged.values()), list(range(1, highest_level + 1))


def normalize_course_id(value: str) -> str:
    """Accept common human formatting such as ``CIVL 2020`` or ``CIVL-2020``."""

    normalized = re.sub(r"[\s-]+", "", value).upper()
    return normalized


def extract_conversation_completed_courses(
    question: str,
    conversation: list[dict[str, str]] | None,
    completed_courses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract explicitly completed courses for this conversation only."""

    merged = {
        normalize_course_id(item["course_id"]): {
            "course_id": normalize_course_id(item["course_id"]),
            "grade": item.get("grade"),
        }
        for item in completed_courses
        if item.get("course_id")
    }
    messages = [
        message.get("content", "") for message in (conversation or [])
        if message.get("role") == "user"
    ] + [question]
    candidates: set[str] = set()
    for text in messages:
        for statement in re.split(r"[.!?\n]+", text):
            if not COMPLETION_CUES.search(statement):
                continue
            if re.search(
                r"\b(?:haven't|have not|hadn't|had not|didn't|did not|not|no)\s+"
                r"(?:completed?|finished|passed|taken|done)\b",
                statement,
                re.IGNORECASE,
            ):
                continue
            candidates.update(
                f"{subject.upper()}{number}"
                for subject, number in COURSE_ID_PATTERN.findall(statement)
            )
    if not candidates:
        return list(merged.values()), []

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT course_id FROM courses WHERE course_id = ANY(%s)",
                (sorted(candidates),),
            )
            recognized = sorted(row[0] for row in cursor.fetchall())
    added = []
    for course_id in recognized:
        if course_id not in merged:
            merged[course_id] = {"course_id": course_id, "grade": None}
            added.append(course_id)
    return list(merged.values()), added


def requested_target_level(question: str) -> int | None:
    """Return the level explicitly requested this turn; history never supplies it."""

    match = LEVEL_NUMBER_PATTERN.search(question)
    if not match or match.group(1) == "&":
        return None
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "uno": 1, "dos": 2,
        "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    }
    value = match.group(1).lower()
    return words.get(value, int(value) if value.isdigit() else None)


def get_student_level_readiness(
    program_id: str, target_level: int, completed_courses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare persisted completion against exact prior-level required courses."""

    prerequisite_level = target_level - 1
    required = get_program_level_courses(program_id, prerequisite_level)
    completed = {normalize_course_id(item["course_id"]) for item in completed_courses}
    missing = [
        course for course in required
        if course["required"] and course["course_id"] not in completed
    ]
    return {
        "target_level": target_level,
        "prerequisite_level": prerequisite_level,
        "ready": not missing,
        "missing_courses": missing,
        "completed_course_count": len(completed),
    }


def hide_internal_program_ids(answer: str) -> str:
    """Prevent database program keys from leaking into student-facing prose."""
    # Do not corrupt a legitimate source URL whose slug happens to contain the
    # internal key (for example, ``...-0816cm/``).
    return re.sub(
        r"(?<![\w/-])\d{4}[A-Z]{2,}(?![\w/-])",
        "the program",
        answer or "",
        flags=re.IGNORECASE,
    )


def finalize_student_answer(answer: str, question: str, tool_results: list[dict[str, Any]]) -> str:
    """Apply deterministic student-facing code, program-ID, and link rules."""
    final = re.sub(
        r"\b(?:your\s+)?academic profile\b",
        "courses you told me about in this conversation",
        answer or "",
        flags=re.IGNORECASE,
    )
    course_records: dict[str, str] = {}
    program_records: dict[str, dict[str, Any]] = {}

    def collect(value: Any):
        if isinstance(value, dict):
            if value.get("course_id") and value.get("course_name"):
                course_records[str(value["course_id"])] = str(value["course_name"])
            if value.get("program_id") and value.get("program_name"):
                program_records[str(value["program_id"])] = value
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(tool_results)
    for course_id, course_name in sorted(course_records.items(), key=lambda item: -len(item[1])):
        if course_name.lower() in final.lower() and course_id.lower() not in final.lower():
            final = re.sub(re.escape(course_name), f"{course_id} {course_name}", final,
                           count=1, flags=re.IGNORECASE)

    if explicitly_requests_program_ids(question):
        missing = [f"- {record['program_name']} — `{program_id}`"
                   for program_id, record in program_records.items()
                   if program_id.lower() not in final.lower()]
        if missing:
            final = final.rstrip() + "\n\n" + "\n".join(missing)
    else:
        final = hide_internal_program_ids(final)

    if explicitly_requests_program_links(question):
        missing = [f"- {record['program_name']} — {record['source_url']}"
                   for record in program_records.values()
                   if record.get("source_url") and str(record["source_url"]) not in final]
        if missing:
            final = final.rstrip() + "\n\n" + "\n".join(missing)
    return final


def _compact_program_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep model context focused while retaining the engine's verified answer."""

    return {
        "program_id": result.get("program_id"),
        "program_complete": result.get("program_complete"),
        "completion_percentage": result.get("completion_percentage"),
        "current_level": result.get("current_level"),
        "evaluation_level": result.get("evaluation_level"),
        "next_level": result.get("next_level"),
        "progression_status": result.get("progression_status"),
        "next_courses": result.get("next_courses", []),
        "missing_requirements": result.get("program_progress", {}).get(
            "missing_requirements", []
        ),
        "unsatisfied_choice_groups": [
            group for group in result.get("program_progress", {}).get("choice_groups", [])
            if not group.get("satisfied")
        ],
        "optional_requirements": result.get("program_progress", {}).get(
            "optional_requirements", []
        ),
    }


def get_student_level_requirements(
    program_id: str, level: int, completed_courses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate core, configured choice groups, and optional items for one level."""
    progress = check_program_progress(
        program_id=program_id,
        completed_courses=completed_courses,
        evaluation_level=level,
    )
    names = {item["course_id"]: item["course_name"]
             for item in get_program_level_courses(program_id, level)}
    missing = [item for item in progress.get("missing_requirements", [])
               if item.get("level") == level]
    for item in missing:
        item["course_name"] = names.get(item["course_id"])
    groups = [group for group in progress.get("choice_groups", [])
              if group.get("level") == level and not group.get("satisfied")]
    for group in groups:
        group["available_options"] = [
            {"course_id": course_id, "course_name": names.get(course_id)}
            for course_id in group.get("available_courses", [])
        ]
    optional_by_id = {
        item["course_id"]: item for item in progress.get("optional_requirements", [])
    }
    optional = []
    for course in get_program_level_courses(program_id, level):
        item = optional_by_id.get(course["course_id"])
        if item:
            item = dict(item, level=level, course_name=course["course_name"])
            optional.append(item)
    return {
        "level": level,
        "missing_required_courses": missing,
        "unsatisfied_choice_groups": groups,
        "optional_requirements": optional,
    }


def execute_tool(name: str, arguments: dict[str, Any], context: dict[str, Any]):
    """Execute only allow-listed Asteris functions with server-owned student data."""

    if name == "search_courses":
        query = arguments["query"]
        normalized = normalize_course_id(query)
        if re.fullmatch(r"[A-Z]{3,5}\d{4}", normalized):
            query = normalized
        return {"courses": find_courses(query)}

    if name == "get_course_details":
        course_id = normalize_course_id(arguments["course_id"])
        course, prerequisites = get_course_details(course_id)
        return {"course": course, "prerequisites": prerequisites}

    if name == "search_programs":
        programs = find_programs(
            "all programs" if context.get("global_catalog") else arguments["query"]
        )
        return {
            "programs": programs,
            "active_program_count": len(programs),
            "distinct_credentials": sorted({
                program["credential"] for program in programs if program.get("credential")
            }),
        }

    if name == "get_program_details":
        resolved_program_id = context.get("program_id") or arguments["program_id"]
        return {"program": get_program_details(resolved_program_id)}

    if name == "get_program_level_courses":
        if not context.get("program_id"):
            return {"error": "A friendly program name is needed to list its courses."}
        return {
            "pathway": context.get("pathway", {}).get("name"),
            "level": arguments["level"],
            "courses": get_program_level_courses(
                context["program_id"], arguments["level"]
            ),
        }

    if name == "get_program_electives":
        if not context.get("program_id"):
            return {"error": "Please name the program whose electives you mean."}
        return {"program_name": context.get("program_name"),
                "level": context.get("level"),
                "choice_groups": get_program_electives(
                    context["program_id"], context.get("level"))}

    if name == "get_program_progression_requirements":
        if not context.get("program_id"):
            return {"error": "Please name the program whose progression rules you mean."}
        requirements = get_program_progression_requirements(context["program_id"])
        details = get_program_details(context["program_id"])
        return {"program_name": context.get("program_name"), "requirements": requirements,
                "source_url": details.get("source_url") if details else None}

    if name == "check_course_eligibility":
        course_id = normalize_course_id(arguments["course_id"])
        prerequisite_result = check_course_eligibility(
            course_id=course_id,
            completed_courses=context["completed_courses"],
            program_id=context.get("program_id"),
        )

        if not context.get("program_id"):
            return {
                "course_id": course_id,
                "course_name": prerequisite_result.get("course_name"),
                "eligible": None,
                "prerequisites_satisfied": prerequisite_result.get("eligible"),
                "missing_requirements": prerequisite_result.get(
                    "missing_requirements", []
                ),
                "message": (
                    "A program ID is required to verify program-level eligibility."
                ),
            }

        advisor_result = run_advisor_engine(
            program_id=context["program_id"],
            completed_courses=context["completed_courses"],
            gpa=context.get("gpa"),
            work_hours=context.get("work_hours", 0),
            diploma_completed=context.get("diploma_completed", False),
        )
        eligible_course = next(
            (
                course
                for course in advisor_result.get("eligible_courses", [])
                if course.get("course_id") == course_id
            ),
            None,
        )
        blocked_course = next(
            (
                course
                for course in advisor_result.get("blocked_courses", [])
                if course.get("course_id") == course_id
            ),
            None,
        )
        course_result = eligible_course or blocked_course or prerequisite_result

        return {
            "course_id": course_id,
            "course_name": course_result.get("course_name"),
            "eligible": eligible_course is not None,
            "current_level": advisor_result.get("current_level"),
            "evaluation_level": advisor_result.get("evaluation_level"),
            "missing_requirements": course_result.get("missing_requirements", []),
        }

    if name == "get_student_program_advice":
        if not context.get("program_id"):
            return {"error": "A program ID is required for program advice."}

        if context.get("level"):
            return get_student_level_requirements(
                context["program_id"], context["level"], context["completed_courses"]
            )
        result = run_advisor_engine(
            program_id=context["program_id"],
            completed_courses=context["completed_courses"],
            gpa=context.get("gpa"),
            work_hours=context.get("work_hours", 0),
            diploma_completed=context.get("diploma_completed", False),
        )
        return _compact_program_result(result)

    if name == "get_student_level_readiness":
        if not context.get("program_id"):
            return {"error": "Please name the program whose level requirements you mean."}
        return get_student_level_readiness(
            context["program_id"], arguments["target_level"], context["completed_courses"]
        )

    return {"error": f"Unknown advisor tool: {name}"}


def answer_student_question(
    question: str,
    completed_courses: list[dict[str, Any]],
    program_id: str | None = None,
    gpa: float | None = None,
    work_hours: float = 0,
    diploma_completed: bool = False,
    conversation: list[dict[str, str]] | None = None,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Let the model select verified tools, then explain their results."""

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    if is_out_of_scope_question(question):
        return {
            "answer": out_of_scope_response(question),
            "tools_used": [],
            "intent": AdvisorIntent.OUT_OF_SCOPE.value,
            "resolved_program": None,
        }

    conversation_courses, extracted_course_ids = extract_conversation_completed_courses(
        question, conversation, completed_courses
    )
    if re.search(r"\b(?:level|nivel)\s*&(?=\s|$|[?.!,])", question, re.IGNORECASE):
        return {
            "answer": "Did you mean Level 7?",
            "tools_used": [],
            "intent": AdvisorIntent.FALLBACK.value,
        }

    intent = classify_intent(question)
    response_language = current_message_language(question)
    resolution = resolve_academic_context(
        question, conversation, program_id.upper() if program_id else None
    )
    if (
        resolution.get("program_id")
        and re.search(r"\b(?:website|webpage|link|url|sitio web|página web|pagina web)\b", question, re.IGNORECASE)
    ):
        intent = AdvisorIntent.PROGRAM_INFO
    effective_program_id = resolution["program_id"]
    effective_courses, completed_level_claims = infer_completed_levels(
        question, effective_program_id, conversation_courses
    )
    context = {
        "program_id": effective_program_id,
        "completed_courses": effective_courses,
        "gpa": gpa,
        "work_hours": work_hours,
        "diploma_completed": diploma_completed,
        "pathway": resolution["pathway"],
        "program_name": resolution.get("program_name"),
        "level": resolution.get("level"),
        "requirement_type": resolution.get("requirement_type"),
        "course_id": resolution.get("course_id"),
        "global_catalog": resolution.get("global_catalog", False),
    }
    api_client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    instructions = (
        f"You are Asteris, an academic advisor for {INSTITUTION_NAME}. "
        f"The question has been routed as {intent.value}. "
        "Use the provided tools for every factual claim about courses, "
        "prerequisites, eligibility, or program progress. Never guess policy. "
        "If no tool can verify a factual or policy claim, say that it cannot be "
        "verified from the available Asteris data. "
        "Internal IDs are implementation details: never ask a student for a program ID. "
        "Course codes are normal student-facing information: whenever a course is "
        "named or returned, show its course_id alongside its course_name. Never redact "
        "a course_id. Program IDs may be omitted in normal prose, but when the current "
        "question explicitly requests program IDs or codes, return the real program_id. "
        "Resolve course and program names with the search tools. Ask for the friendly "
        "program name only when eligibility or progress truly requires it. "
        "Treat the Civil Engineering Diploma and Bachelor of Engineering as related "
        "parts of one pathway. When a requested level is shared, list it directly "
        "instead of asking which credential the student means. An explicit statement "
        "that all courses through a level were completed is sufficient completion "
        "context for a progress calculation; do not request an official outline. "
        "For course information, describe the course and its prerequisites without "
        "discussing completed-course state or eligibility. For program information, "
        "aggregate the curated program record, its configured program courses, and "
        "its stored source URL. Category questions must search programs "
        "by credential. An explicit whole-catalog question overrides all prior or "
        "selected program context; report its count, program list, or distinct "
        "credentials only from the search_programs result. Electives come only from "
        "choice groups: required=false does "
        "not mean elective, and work terms are not electives. Practical-work and "
        "progression answers come only from progression requirements. Progress "
        "percentages must be quoted only from the program-progress tool and never "
        "calculated from level numbers. If requested detail is missing and a verified "
        "source_url exists, provide that link as the fallback. "
        "Do not claim that an answer is official; encourage confirmation with "
        "the institution when data is missing or ambiguous. Be concise and clear. "
        "Before saying a verified requirement is unavailable, call every relevant "
        "deterministic tool provided for the routed intent. Course completion state "
        "comes only from explicit statements by the student in the current conversation. "
        "Describe it as courses the student told you they completed; never mention or "
        "imply an academic profile or stored student record. Spanish "
        "uses exactly the same tools and rules as English. Respond entirely in the "
        f"language of the current student question: {response_language}. Conversation "
        "history may provide subject or entity context but must never determine or "
        "carry over the response language. Keep co-op work terms, practical-work hours, and academic "
        "level progression separate. A level explicitly named in the current question "
        "overrides unrelated levels in history; never offer progression to another level."
    )
    history_lines = []
    for message in (conversation or [])[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = message.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role.title()}: {content}")

    history = "\n".join(history_lines)
    if len(history) > MAX_HISTORY_CHARACTERS:
        history = history[-MAX_HISTORY_CHARACTERS:]

    initial_input = (
        f"Student question: {question}\n"
        f"Routed intent: {intent.value}\n"
        f"Required response language from current turn: {response_language}"
    )
    if intent in {AdvisorIntent.ELIGIBILITY_NEXT_COURSES, AdvisorIntent.PROGRAM_PROGRESS}:
        initial_input += (
            f"\nKnown internal program: {context['program_id'] or 'not selected'}"
            f"\nCompleted course count: {len(effective_courses)}"
        )
    if resolution["pathway"]:
        initial_input += "\nResolved academic pathway: Civil Engineering"
    if intent == AdvisorIntent.PROGRAM_INFO and effective_program_id:
        initial_input += (
            f"\nResolved internal program for tool use: {effective_program_id}. "
            "Use get_program_details so the answer includes the curated program "
            "record, related courses, and stored source URL."
        )
    if completed_level_claims:
        initial_input += (
            "\nStudent explicitly reported completing all required courses through "
            f"Level {max(completed_level_claims)}; those curated required courses "
            "have been supplied to the progress engine."
        )
    target_level = requested_target_level(question)
    if target_level:
        initial_input += (
            f"\nCurrent-turn target level: {target_level} "
            "(do not substitute a historical level)."
        )
    if extracted_course_ids:
        initial_input += (
            "\nCourses the student explicitly said they completed in this conversation: "
            + ", ".join(extracted_course_ids)
        )
    if history:
        initial_input += f"\nRecent conversation:\n{history}"
    if resolution.get("global_catalog"):
        initial_input += (
            "\nThis is an explicit whole-catalog request. Ignore program-specific "
            "history and use the complete active catalog returned by search_programs."
        )
    if explicitly_requests_program_ids(question):
        initial_input += "\nExplicit ID request: include each real program_id from the tool result."
    if explicitly_requests_program_links(question):
        initial_input += "\nExplicit link request: include each stored source_url; report null URLs as missing data."

    allowed_names = TOOLS_BY_INTENT[intent]
    allowed_tools = [tool for tool in TOOLS if tool["name"] in allowed_names]
    request_options = {
        "model": MODEL,
        "instructions": instructions,
        "input": initial_input,
        "tools": allowed_tools,
    }
    if intent == AdvisorIntent.ELECTIVES and effective_program_id:
        request_options["tool_choice"] = {
            "type": "function", "name": "get_program_electives"
        }
    elif resolution.get("global_catalog"):
        request_options["tool_choice"] = {
            "type": "function", "name": "search_programs"
        }
    elif intent == AdvisorIntent.PROGRAM_INFO and effective_program_id:
        request_options["tool_choice"] = {
            "type": "function", "name": "get_program_details"
        }
    elif target_level and intent == AdvisorIntent.PROGRAM_PROGRESS and effective_program_id:
        request_options["tool_choice"] = {
            "type": "function", "name": "get_student_program_advice"
        }
    elif intent == AdvisorIntent.PROGRAM_PROGRESS and effective_program_id:
        request_options["tool_choice"] = "required"
    response = api_client.responses.create(**request_options)
    tools_used = []
    tool_results = []

    for _ in range(MAX_TOOL_ROUNDS):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return {
                "answer": finalize_student_answer(response.output_text, question, tool_results),
                "tools_used": tools_used,
                "intent": intent.value,
                "resolved_program": context.get("program_name"),
            }

        outputs = []
        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
                result = execute_tool(call.name, arguments, context)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                result = {"error": f"Invalid tool request: {error}"}

            tools_used.append(call.name)
            tool_results.append(result)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                }
            )

        response = api_client.responses.create(
            model=MODEL,
            instructions=instructions,
            previous_response_id=response.id,
            input=outputs,
            tools=allowed_tools,
        )

    return {
        "answer": (
            "I could not complete that advising lookup. Please narrow the question "
            "or contact an academic advisor."
        ),
        "tools_used": tools_used,
        "intent": intent.value,
        "error": "tool_round_limit_reached",
    }
