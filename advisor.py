import os
import re

from fastapi import HTTPException
from openai import OpenAI

from database import get_connection
from eligibility import check_course_eligibility
from advisor_engine import run_advisor_engine


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_course_code(question: str):
    match = re.search(r"\b[A-Z]{3,5}\d{4}\b", question.upper())

    if match:
        return match.group(0)

    return None


def find_courses(search_text: str):
    search_text = search_text.strip()

    if not search_text:
        return []

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    course_id,
                    course_name
                FROM courses
                WHERE status = 'Active'
                  AND (
                      course_id ILIKE %s
                      OR course_name ILIKE %s
                  )
                ORDER BY
                    CASE
                        WHEN UPPER(course_id) = UPPER(%s) THEN 0
                        WHEN LOWER(course_name) = LOWER(%s) THEN 1
                        ELSE 2
                    END,
                    course_id
                LIMIT 10
                """,
                (
                    f"%{search_text}%",
                    f"%{search_text}%",
                    search_text,
                    search_text,
                ),
            )

            rows = cursor.fetchall()

    return [
        {
            "course_id": row[0],
            "course_name": row[1],
        }
        for row in rows
    ]


def get_course_details(course_id: str):
    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    course_id,
                    course_name,
                    credits,
                    course_overview,
                    status,
                    source_url,
                    last_checked,
                    notes
                FROM courses
                WHERE course_id = %s
                """,
                (course_id.upper(),),
            )

            course = cursor.fetchone()

            if course is None:
                return None, []

            cursor.execute(
                """
                SELECT
                    pg.group_type,
                    pc.prerequisite_course_id,
                    pc.minimum_grade,
                    pc.required_program_id,
                    pc.condition_type,
                    pc.notes
                FROM prerequisite_groups pg
                JOIN prerequisite_conditions pc
                    ON pc.prerequisite_group_id =
                       pg.prerequisite_group_id
                WHERE pg.course_id = %s
                ORDER BY
                    pg.prerequisite_group_id,
                    pc.prerequisite_condition_id
                """,
                (course_id.upper(),),
            )

            prerequisite_rows = cursor.fetchall()

    course_data = {
        "course_id": course[0],
        "course_name": course[1],
        "credits": course[2],
        "course_overview": course[3],
        "status": course[4],
        "source_url": course[5],
        "last_checked": str(course[6]) if course[6] else None,
        "notes": course[7],
    }

    prerequisite_data = [
        {
            "group_type": row[0],
            "course_id": row[1],
            "minimum_grade": row[2],
            "required_program_id": row[3],
            "condition_type": row[4],
            "notes": row[5],
        }
        for row in prerequisite_rows
    ]

    return course_data, prerequisite_data


def is_next_courses_question(question: str):
    question_lower = question.lower()

    next_course_phrases = [
        "what courses can i take next",
        "what can i take next",
        "which courses can i take next",
        "what courses should i take next",
        "what should i take next",
        "what am i eligible for",
        "what courses am i eligible for",
        "which courses am i eligible for",
        "what courses can i take",
        "which courses can i take",
    ]

    return any(
        phrase in question_lower
        for phrase in next_course_phrases
    )


def advisor_course_question(
    question: str,
    completed_courses: list,
    program_id: str | None = None,
    gpa=None,
    work_hours=0,
    diploma_completed=False,
):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    # ---------------------------------------------------------
    # NEXT-COURSES / ELIGIBILITY QUESTIONS
    # ---------------------------------------------------------

    if is_next_courses_question(question):

        if not program_id:
            return {
                "found": False,
                "question": question,
                "message": (
                    "Please provide your BCIT program so I can "
                    "determine which courses you can take next."
                ),
            }

        advisor_result = run_advisor_engine(
    program_id=program_id,
    completed_courses=completed_courses,
    gpa=gpa,
    work_hours=work_hours,
    diploma_completed=diploma_completed,
)

        answer_response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=(
                "You are Asteris, a BCIT academic information assistant. "
                "The advisor engine has already calculated the student's "
                "verified eligible courses. "
                "Use ONLY the supplied advisor results. "
                "Do not invent courses, prerequisites, grades, policies, "
                "or requirements. "
                "Clearly list the courses the student can take next. "
                "Keep the answer concise and student-friendly."
            ),
            input=f"""
Student question:
{question}

Program:
{program_id}

Completed courses:
{completed_courses}

Verified advisor engine result:
{advisor_result}
""",
        )

        return {
            "found": True,
            "question": question,
            "type": "next_courses",
            "program_id": program_id,
            "advisor_result": advisor_result,
            "answer": answer_response.output_text,
        }

    # ---------------------------------------------------------
    # INDIVIDUAL COURSE QUESTIONS
    # ---------------------------------------------------------

    course_code = extract_course_code(question)
    matches = find_courses(course_code or question)

    if not matches:
        return {
            "found": False,
            "question": question,
            "message": (
                "I could not find a matching active BCIT course. "
                "Please include the course name or course code."
            ),
        }

    if len(matches) > 1:
        return {
            "found": True,
            "multiple_matches": True,
            "question": question,
            "courses": matches,
            "message": (
                "I found multiple matching courses. "
                "Please specify the course code or full course name."
            ),
        }

    course = matches[0]

    course_data, prerequisite_data = get_course_details(
        course["course_id"]
    )

    eligibility_result = check_course_eligibility(
        course_id=course["course_id"],
        completed_courses=completed_courses,
        program_id=program_id,
    )

    answer_response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=(
            "You are Asteris, a BCIT academic information assistant. "
            "The database and rules engine have already determined the "
            "verified result. Explain that result clearly and concisely. "
            "Use ONLY the supplied information. "
            "Do not invent requirements, courses, grades, policies, "
            "or eligibility rules. "
            "If the student is not eligible, clearly explain what is "
            "missing. If the student is eligible, clearly explain why."
        ),
        input=f"""
Student question:
{question}

Course:
{course_data}

Prerequisites:
{prerequisite_data}

Verified eligibility result:
{eligibility_result}
""",
    )

    return {
        "found": True,
        "multiple_matches": False,
        "course": course_data,
        "eligibility": eligibility_result,
        "answer": answer_response.output_text,
    }