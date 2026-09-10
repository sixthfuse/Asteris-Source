from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import get_connection
from eligibility import check_course_eligibility
from program_requirements import check_program_progress
from progression import check_progression
from ai_advisor import answer_student_question
from advisor_eligibility import find_eligible_courses
from advisor_engine import run_advisor_engine
from advisor_v4 import answer_student_question_v4

app = FastAPI(title="Asteris API")
app.mount("/static", StaticFiles(directory="static"), name="static")

class AskRequest(BaseModel):
    question: str


class CompletedCourse(BaseModel):
    course_id: str
    grade: float | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EligibilityRequest(BaseModel):
    completed_courses: list[CompletedCourse]
    program_id: str | None = None
    gpa: float | None = None
    work_hours: float = 0
    diploma_completed: bool = False

class AdvisorRequest(BaseModel):
    question: str
    conversation: list[ChatMessage] = Field(default_factory=list)
    advisor_state: dict | None = None


@app.get("/")
def root():
    return {"message": "Asteris API is running"}


@app.get("/app", include_in_schema=False)
def student_app():
    return FileResponse("static/index.html")


@app.get("/courses/count")
def course_count():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM courses")
            count = cursor.fetchone()[0]

    return {"course_count": count}


@app.get("/programs")
def list_programs():
    """Return friendly choices for the student interface."""

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
            rows = cursor.fetchall()
    return {
        "programs": [
            {"program_id": row[0], "program_name": row[1], "credential": row[2]}
            for row in rows
        ]
    }


@app.get("/courses/{course_id}")
def get_course(course_id: str):

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
        raise HTTPException(
            status_code=404,
            detail="Course not found",
        )

    return {
        "course_id": course[0],
        "course_name": course[1],
        "credits": course[2],
        "course_overview": course[3],
        "status": course[4],
        "source_url": course[5],
        "last_checked": course[6],
        "notes": course[7],
    }


@app.get("/courses/{course_id}/prerequisites")
def get_prerequisites(course_id: str):

    with get_connection() as connection:
        with connection.cursor() as cursor:
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
                ORDER BY pg.prerequisite_group_id,
                         pc.prerequisite_condition_id
                """,
                (course_id.upper(),),
            )

            rows = cursor.fetchall()

    return {
        "course_id": course_id.upper(),
        "prerequisites": [
            {
                "group_type": row[0],
                "prerequisite_course_id": row[1],
                "minimum_grade": row[2],
                "required_program_id": row[3],
                "condition_type": row[4],
                "notes": row[5],
            }
            for row in rows
        ],
    }


@app.post("/eligibility/{course_id}")
def eligibility(
    course_id: str,
    request: EligibilityRequest,
):

    completed_courses = [
        {
            "course_id": item.course_id,
            "grade": item.grade,
        }
        for item in request.completed_courses
    ]

    return check_course_eligibility(
        course_id=course_id,
        completed_courses=completed_courses,
        program_id=request.program_id,
    )

@app.post("/program/{program_id}/eligible-courses")
def eligible_courses(
    program_id: str,
    request: EligibilityRequest,
):
    completed_courses = [
        {
            "course_id": item.course_id,
            "grade": item.grade,
        }
        for item in request.completed_courses
    ]

    return find_eligible_courses(
        program_id=program_id,
        completed_courses=completed_courses,
    )

@app.get("/program/{program_id}/courses")
def get_program_courses(program_id: str):

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    pc.course_id,
                    c.course_name,
                    c.credits,
                    c.source_url,
                    pc.level,
                    pc.term,
                    pc.course_type,
                    pc.required,
                    pc.notes
                FROM program_courses pc
                JOIN courses c
                    ON c.course_id = pc.course_id
                WHERE pc.program_id = %s
                ORDER BY pc.level, pc.course_id
                """,
                (program_id.upper(),),
            )

            rows = cursor.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Program courses not found",
        )

    return {
        "program_id": program_id.upper(),
        "course_count": len(rows),
        "courses": [
            {
                "course_id": row[0],
                "course_name": row[1],
                "credits": row[2],
                "source_url": row[3],
                "level": row[4],
                "term": row[5],
                "course_type": row[6],
                "required": row[7],
                "notes": row[8],
            }
            for row in rows
        ],
    }


@app.post("/program/{program_id}/progress")
def program_progress(
    program_id: str,
    request: EligibilityRequest,
):
    completed_courses = [
        {
            "course_id": item.course_id,
            "grade": item.grade,
        }
        for item in request.completed_courses
    ]

    return check_program_progress(
        program_id=program_id,
        completed_courses=completed_courses,
    )


@app.post("/program/{program_id}/progression/{from_level}")
def progression(
    program_id: str,
    from_level: int,
    request: EligibilityRequest,
):
    completed_courses = [
        {
            "course_id": item.course_id,
            "grade": item.grade,
        }
        for item in request.completed_courses
    ]

    return check_progression(
        program_id=program_id,
        from_level=from_level,
        completed_courses=completed_courses,
        gpa=request.gpa,
        work_hours=request.work_hours,
        diploma_completed=request.diploma_completed,
    )


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_student_question(
            question=request.question,
            completed_courses=[],
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/advisor")
def advisor(request: AdvisorRequest):
    try:
        return answer_student_question(
            question=request.question,
            completed_courses=[],
            conversation=[
                message.model_dump()
                for message in request.conversation
            ],
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/advisor-v4")
def advisor_v4(request: AdvisorRequest):
    try:
        return answer_student_question_v4(
            question=request.question,
            conversation=[
                message.model_dump()
                for message in request.conversation
            ],
            advisor_state=request.advisor_state,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/advisor/engine")
def advisor_engine(request: EligibilityRequest):

    completed_courses = [
        {
            "course_id": item.course_id,
            "grade": item.grade,
        }
        for item in request.completed_courses
    ]

    return run_advisor_engine(
        program_id=request.program_id,
        completed_courses=completed_courses,
        gpa=request.gpa,
        work_hours=request.work_hours,
        diploma_completed=request.diploma_completed,
    )
