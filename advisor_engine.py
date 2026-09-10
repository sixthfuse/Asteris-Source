from database import get_connection
from advisor_eligibility import find_eligible_courses
from program_requirements import check_program_progress
from progression import check_progression


def get_program_levels(program_id):
    """
    Return all program levels that contain courses.

    Levels are determined directly from program_courses.
    This allows the advisor engine to know which level is the
    final level of the program.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT level
                FROM program_courses
                WHERE program_id = %s
                  AND level IS NOT NULL
                ORDER BY level
                """,
                (program_id,),
            )

            rows = cursor.fetchall()

    return [row[0] for row in rows]


def get_required_courses_by_level(program_id):
    """
    Return required program courses grouped by level.

    Work terms and elective choices that are not marked required
    are intentionally excluded from this calculation.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    level,
                    course_id
                FROM program_courses
                WHERE program_id = %s
                  AND required = TRUE
                  AND level IS NOT NULL
                ORDER BY level, course_id
                """,
                (program_id,),
            )

            rows = cursor.fetchall()

    required_by_level = {}

    for level, course_id in rows:
        required_by_level.setdefault(level, set()).add(
            course_id.upper()
        )

    return required_by_level


def determine_current_level(
    required_by_level,
    completed_ids,
):
    """
    Determine the highest level for which all required courses
    have been completed.

    Example:

        Level 1 complete
        Level 2 complete
        Level 3 complete
        Level 4 complete
        Level 5 complete
        Level 6 complete
        Level 7 complete
        Level 8 not complete

    returns 7.

    This represents the highest fully completed required-course
    level. The advisor engine can then decide whether Level 8
    should be evaluated.
    """

    current_level = 0

    for level in sorted(required_by_level):

        required_ids = required_by_level[level]

        if required_ids.issubset(completed_ids):
            current_level = level
        else:
            break

    return current_level


def run_advisor_engine(
    program_id,
    completed_courses,
    gpa=None,
    work_hours=0,
    diploma_completed=False,
):
    """
    Main Asteris advisor engine.

    Level terminology:

        current_level
            Last fully completed academic level.

        evaluation_level
            Level whose courses are currently being evaluated.

        next_level
            Level the student is entering next.

        final_program_level
            Highest level in the program.

    The coordinator determines these levels once and passes
    them explicitly into the program-progress layer.

    The final program level may be evaluated once the student
    has completed the preceding level, even if no separate
    progression requirement exists for entry to that final level.
    """

    if not program_id:
        return {
            "error": "Program ID is required"
        }

    program_id = program_id.upper()

    completed_courses = [
        {
            "course_id": item["course_id"].upper(),
            "grade": item.get("grade"),
        }
        for item in completed_courses
    ]

    completed_ids = {
        item["course_id"]
        for item in completed_courses
    }

    # ---------------------------------------------------------
    # PROGRAM LEVELS
    # ---------------------------------------------------------

    all_program_levels = get_program_levels(program_id)

    if not all_program_levels:
        return {
            "program_id": program_id,
            "error": "Program levels not found",
        }

    final_program_level = max(all_program_levels)

    required_by_level = get_required_courses_by_level(
        program_id
    )

    # ---------------------------------------------------------
    # DETERMINE CURRENT LEVEL
    # ---------------------------------------------------------
    #
    # current_level means the LAST COMPLETED academic level.
    #
    # It must never be relabeled as the level currently being
    # evaluated.
    # ---------------------------------------------------------

    current_level = determine_current_level(
        required_by_level=required_by_level,
        completed_ids=completed_ids,
    )

    # ---------------------------------------------------------
    # DETERMINE CANDIDATE NEXT LEVEL
    # ---------------------------------------------------------

    if current_level < final_program_level:
        candidate_next_level = current_level + 1
    else:
        candidate_next_level = final_program_level

    # ---------------------------------------------------------
    # DETERMINE EVALUATION LEVEL
    # ---------------------------------------------------------
    #
    # The coordinator owns this decision.
    #
    # Normal progression:
    #
    #     approved -> evaluate next level
    #
    # Final-level entry:
    #
    #     no separate progression gate is required -> evaluate
    #     the final program level
    #
    # Initial student:
    #
    #     evaluate the first program level.
    # ---------------------------------------------------------

    final_level_entry = (
        current_level < final_program_level
        and candidate_next_level == final_program_level
    )

    # We initially assume the current level is being evaluated.
    evaluation_level = current_level

    # ---------------------------------------------------------
    # PROGRESSION
    # ---------------------------------------------------------

    progression_result = None

    if (
        current_level > 0
        and current_level < final_program_level
    ):

        progression_result = check_progression(
            program_id=program_id,
            from_level=current_level,
            completed_courses=completed_courses,
            gpa=gpa,
            work_hours=work_hours,
            diploma_completed=diploma_completed,
        )

        # -----------------------------------------------------
        # FINAL LEVEL WITH NO SEPARATE PROGRESSION GATE
        #
        # A missing 7 -> 8 progression rule is intentional.
        # It does NOT mean the student is blocked or that the
        # progression definition is broken.
        # -----------------------------------------------------

        if progression_result.get("error") == (
            "Progression requirements not found"
        ):

            if final_level_entry:

                progression_result = {
                    "program_id": program_id,
                    "from_level": current_level,
                    "to_level": candidate_next_level,
                    "eligible": True,
                    "status": "NOT_REQUIRED",
                    "message": (
                        "No separate progression requirements "
                        "apply for entry to the final level."
                    ),
                }

            else:

                progression_result = {
                    "program_id": program_id,
                    "from_level": current_level,
                    "to_level": candidate_next_level,
                    "eligible": True,
                    "status": "NOT_REQUIRED",
                    "message": (
                        "No separate progression requirements "
			"apply for this level transition."
                    ),
                }

    # ---------------------------------------------------------
    # FINAL LEVEL / NORMAL PROGRESSION DECISION
    # ---------------------------------------------------------

    progression_approved = (
        progression_result is not None
        and progression_result.get("eligible") is True
    )

    if progression_approved:
        evaluation_level = candidate_next_level

    elif final_level_entry:
        evaluation_level = final_program_level

    elif current_level == 0:
        evaluation_level = min(all_program_levels)

    # Student already at final level.
    if current_level == final_program_level:
        evaluation_level = final_program_level

    # ---------------------------------------------------------
    # PROGRAM PROGRESS
    # ---------------------------------------------------------
    #
    # Pass the coordinator's decisions explicitly.
    #
    # program_requirements.py must not infer or relabel
    # current_level.
    # ---------------------------------------------------------

    progress_result = check_program_progress(
        program_id=program_id,
        completed_courses=completed_courses,
        current_level=current_level,
        evaluation_level=evaluation_level,
    )

    # ---------------------------------------------------------
    # FIND ELIGIBLE COURSES
    # ---------------------------------------------------------

    eligibility_result = find_eligible_courses(
        program_id=program_id,
        completed_courses=completed_courses,
        max_level=evaluation_level,
    )

    eligible_courses = eligibility_result.get(
        "eligible_courses",
        []
    )

    blocked_courses = eligibility_result.get(
        "not_eligible_courses",
        []
    )

    completed_courses_result = eligibility_result.get(
        "completed_courses",
        []
    )

    # ---------------------------------------------------------
    # NEXT LEVEL
    # ---------------------------------------------------------
    #
    # evaluation_level is the level currently being evaluated.
    # If it is above current_level, it is the next level.
    # ---------------------------------------------------------

    if evaluation_level > current_level:
        next_level = evaluation_level
    else:
        next_level = current_level

    if current_level >= final_program_level:
        next_level = final_program_level

    # ---------------------------------------------------------
    # NEXT COURSES
    # ---------------------------------------------------------

    next_courses = [
        course
        for course in eligible_courses
        if course.get("level") == next_level
    ]

    # ---------------------------------------------------------
    # FUTURE COURSES
    # ---------------------------------------------------------

    future_eligible_courses = [
        course
        for course in eligible_courses
        if course.get("level") > next_level
    ]

    # ---------------------------------------------------------
    # RESPONSE SEMANTICS
    # ---------------------------------------------------------

    program_complete = progress_result.get("eligible")

    completion_percentage = progress_result.get(
        "completion_percentage"
    )

    if current_level == 0:

        progression_status = "INITIAL_LEVEL"

    elif final_level_entry:

        progression_status = "FINAL_LEVEL_ENTRY"

    elif progression_result:

        progression_status = progression_result.get(
            "status"
        )

    else:

        progression_status = None

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    summary = {
        "current_level": current_level,
        "evaluation_level": evaluation_level,
        "next_level": next_level,
        "final_program_level": final_program_level,
        "completed_course_count": len(
            completed_courses_result
        ),
        "next_course_count": len(next_courses),
        "next_courses": [
            {
                "course_id": course.get("course_id"),
                "course_name": course.get("course_name"),
                "term": course.get("term"),
                "course_type": course.get("course_type"),
                "required": course.get("required"),
            }
            for course in next_courses
        ],
        "future_eligible_course_count": len(
            future_eligible_courses
        ),
        "progression_status": progression_status,
    }

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    return {
        "program_id": program_id,

        # Retained for compatibility. This means the program
        # is complete, not that every listed course is eligible.
        "eligible": program_complete,

        "program_complete": program_complete,

        "completion_percentage": completion_percentage,

        "current_level": current_level,

        "evaluation_level": evaluation_level,

        "final_program_level": final_program_level,

        "next_level": next_level,

        "progression_status": progression_status,

        "eligible_courses": eligible_courses,

        "next_courses": next_courses,

        "future_eligible_courses": future_eligible_courses,

        "completed_courses": completed_courses_result,

        "blocked_courses": blocked_courses,

        "program_progress": progress_result,

        "progression": progression_result,

        "summary": summary,
    }