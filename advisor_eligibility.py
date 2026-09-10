from database import get_connection
from eligibility import check_course_eligibility


def find_eligible_courses(
    program_id,
    completed_courses,
    max_level=None,
):
    program_id = program_id.upper()

    completed_ids = {
        item["course_id"].upper()
        for item in completed_courses
    }

    # ---------------------------------------------------------
    # GET ALL COURSES BELONGING TO THE PROGRAM
    # ---------------------------------------------------------

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    pc.course_id,
                    c.course_name,
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
                (program_id,),
            )

            courses = cursor.fetchall()

    if not courses:
        return {
            "program_id": program_id,
            "error": "Program courses not found",
        }

    # ---------------------------------------------------------
    # DETERMINE HIGHEST PROGRAM LEVEL
    # ---------------------------------------------------------

    all_program_levels = sorted(
        {
            course[2]
            for course in courses
            if course[2] is not None
        }
    )

    if not all_program_levels:
        return {
            "program_id": program_id,
            "error": "Program levels not found",
        }

    final_program_level = max(all_program_levels)

    # ---------------------------------------------------------
    # DEFAULT EVALUATION LEVEL
    # ---------------------------------------------------------
    #
    # The caller may explicitly provide max_level.
    #
    # If it doesn't, default to the final level rather than
    # trying to independently calculate academic progression.
    #
    # The advisor engine is responsible for deciding what level
    # should actually be evaluated.
    # ---------------------------------------------------------

    if max_level is None:
        max_level = final_program_level

    # Never allow evaluation beyond the program's actual
    # highest level.
    max_level = min(
        max_level,
        final_program_level,
    )

    eligible_courses = []
    not_eligible_courses = []
    completed_program_courses = []

    # ---------------------------------------------------------
    # EVALUATE COURSES
    # ---------------------------------------------------------

    for course in courses:

        (
            course_id,
            course_name,
            level,
            term,
            course_type,
            required,
            notes,
        ) = course

        course_id = course_id.upper()

        # -----------------------------------------------------
        # ALREADY COMPLETED
        # -----------------------------------------------------

        if course_id in completed_ids:

            completed_program_courses.append(
                {
                    "course_id": course_id,
                    "course_name": course_name,
                    "level": level,
                    "term": term,
                    "course_type": course_type,
                    "required": required,
                    "notes": notes,
                }
            )

            continue

        # -----------------------------------------------------
        # COURSE IS ABOVE CURRENTLY ALLOWED LEVEL
        # -----------------------------------------------------

        if level > max_level:

            not_eligible_courses.append(
                {
                    "course_id": course_id,
                    "course_name": course_name,
                    "level": level,
                    "term": term,
                    "course_type": course_type,
                    "required": required,
                    "notes": notes,
                    "missing_requirements": [
                        {
                            "type": "PROGRAM_LEVEL",
                            "required_level": level,
                            "current_level": max_level,
                            "notes": (
                                "Course belongs to a future "
                                "program level."
                            ),
                        }
                    ],
                }
            )

            continue

        # -----------------------------------------------------
        # CHECK COURSE PREREQUISITES
        # -----------------------------------------------------

        result = check_course_eligibility(
            course_id=course_id,
            completed_courses=completed_courses,
            program_id=program_id,
        )

        course_result = {
            "course_id": course_id,
            "course_name": course_name,
            "level": level,
            "term": term,
            "course_type": course_type,
            "required": required,
            "notes": notes,
            "missing_requirements": result.get(
                "missing_requirements",
                [],
            ),
        }

        if result.get("eligible") is True:

            eligible_courses.append(
                course_result
            )

        else:

            not_eligible_courses.append(
                course_result
            )

    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------

    return {
        "program_id": program_id,
        "current_level": max_level,
        "final_program_level": final_program_level,
        "completed_courses": completed_program_courses,
        "eligible_courses": eligible_courses,
        "not_eligible_courses": not_eligible_courses,
        "eligible_count": len(eligible_courses),
        "not_eligible_count": len(
            not_eligible_courses
        ),
        "completed_count": len(
            completed_program_courses
        ),
    }