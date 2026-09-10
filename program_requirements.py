from database import get_connection


def check_program_progress(
    program_id,
    completed_courses,
    current_level=None,
    evaluation_level=None,
):
    """
    Evaluate a student's progress against program requirements and
    required program courses.

    This function handles:

    - Individual course requirements
    - Grouped elective requirements
    - Required number of courses within each choice group
    - Optional co-op work terms
    - Coordinator-supplied current and evaluation levels
    - Final program level
    - Overall completion percentage

    Course eligibility/prerequisite logic is handled separately by
    advisor_eligibility.py.  When the advisor coordinator supplies level
    values, they are authoritative.  The fallback is retained for callers
    that use this module directly.
    """

    program_id = program_id.upper()

    # ---------------------------------------------------------
    # Normalize completed courses
    # ---------------------------------------------------------

    completed = {
        item["course_id"].upper()
        for item in completed_courses
        if item.get("course_id")
    }

    # ---------------------------------------------------------
    # Get program requirements
    # ---------------------------------------------------------

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    program_id,
                    course_id,
                    requirement_type,
                    level,
                    choice_group,
                    choice_required,
                    notes
                FROM program_requirements
                WHERE program_id = %s
                ORDER BY level, course_id
                """,
                (program_id,),
            )

            requirement_rows = cursor.fetchall()

            # -------------------------------------------------
            # Get all courses belonging to the program.
            #
            # This is used to determine the student's current
            # level and the program's final level.
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT
                    course_id,
                    level,
                    required
                FROM program_courses
                WHERE program_id = %s
                ORDER BY level, course_id
                """,
                (program_id,),
            )

            program_course_rows = cursor.fetchall()

    # ---------------------------------------------------------
    # Validate program requirements
    # ---------------------------------------------------------

    if not requirement_rows and not program_course_rows:
        return {
            "program_id": program_id,
            "eligible": False,
            "completion_percentage": 0,
            "completed_courses": sorted(completed),
            "missing_requirements": [],
            "choice_groups": [],
            "optional_requirements": [],
            "current_level": None,
            "evaluation_level": None,
            "final_program_level": None,
            "error": "Program requirements not found",
        }

    # ---------------------------------------------------------
    # Determine program levels
    # ---------------------------------------------------------

    program_levels = sorted(
        {
            row[1]
            for row in program_course_rows
            if row[1] is not None
        }
    )

    if program_levels:
        final_program_level = max(program_levels)
    else:
        final_program_level = None

    # ---------------------------------------------------------
    # Determine current program level only when the coordinator
    # has not supplied one.
    #
    # The fallback matches the coordinator's semantics: the highest
    # consecutive level whose required courses are all complete.  It must
    # not use the first incomplete program course, because that would label
    # a student entering Level 8 as currently being at Level 8.
    # ---------------------------------------------------------

    if current_level is None:
        required_by_level = {}

        for course_id, level, required in program_course_rows:
            if course_id is not None and required and level is not None:
                required_by_level.setdefault(level, set()).add(
                    course_id.upper()
                )

        current_level = 0

        for level in sorted(required_by_level):
            if required_by_level[level].issubset(completed):
                current_level = level
            else:
                break

    # ---------------------------------------------------------
    # Evaluation level
    #
    # The coordinator decides this independently of program completion.
    # Direct callers continue to evaluate the final program level.
    # ---------------------------------------------------------

    if evaluation_level is None:
        evaluation_level = final_program_level

    # ---------------------------------------------------------
    # Separate requirements
    # ---------------------------------------------------------

    individual_requirements = []
    choice_groups = []
    optional_requirements = []
    configured_requirement_ids = set()

    for row in requirement_rows:

        (
            row_program_id,
            course_id,
            requirement_type,
            level,
            choice_group,
            choice_required,
            notes,
        ) = row

        # -----------------------------------------------------
        # Some program_requirements rows may not represent a
        # specific course. Ignore empty course IDs safely.
        # -----------------------------------------------------

        if course_id is None:
            continue

        course_id = course_id.upper()
        configured_requirement_ids.add(course_id)

        # -----------------------------------------------------
        # Work terms are optional because they apply only to
        # students in the cooperative education stream.
        # -----------------------------------------------------

        if requirement_type == "Work Term":

            optional_requirements.append(
                {
                    "course_id": course_id,
                    "requirement_type": requirement_type,
                    "level": level,
                    "completed": course_id in completed,
                    "notes": notes,
                }
            )

            continue

        # -----------------------------------------------------
        # Grouped elective / choice requirement
        # -----------------------------------------------------

        if choice_group:

            existing_group = next(
                (
                    group
                    for group in choice_groups
                    if group["choice_group"] == choice_group
                ),
                None,
            )

            if existing_group is None:

                existing_group = {
                    "choice_group": choice_group,
                    "required": choice_required or 0,
                    "courses": [],
                    "level": level,
                    "requirement_type": requirement_type,
                    "notes": notes,
                }

                choice_groups.append(existing_group)

            existing_group["courses"].append(course_id)

        # -----------------------------------------------------
        # Individual mandatory requirement
        # -----------------------------------------------------

        else:

            individual_requirements.append(
                {
                    "course_id": course_id,
                    "requirement_type": requirement_type,
                    "level": level,
                    "completed": course_id in completed,
                    "notes": notes,
                }
            )

    # ---------------------------------------------------------
    # Add required core courses from program_courses.
    #
    # program_requirements defines elective and co-op rules for
    # this program, while program_courses is the authoritative
    # source for its required core courses. A required course that
    # is not already represented by a program-requirements row is
    # therefore an individual graduation requirement.
    # ---------------------------------------------------------

    for course_id, level, required in program_course_rows:

        if (
            course_id is not None
            and required
            and course_id.upper() not in configured_requirement_ids
        ):

            course_id = course_id.upper()

            individual_requirements.append(
                {
                    "course_id": course_id,
                    "requirement_type": "Core",
                    "level": level,
                    "completed": course_id in completed,
                    "notes": None,
                }
            )

    # ---------------------------------------------------------
    # Process individual requirements
    # ---------------------------------------------------------

    missing_individual = [
        requirement
        for requirement in individual_requirements
        if not requirement["completed"]
    ]

    completed_individual = [
        requirement
        for requirement in individual_requirements
        if requirement["completed"]
    ]

    # ---------------------------------------------------------
    # Process choice groups
    # ---------------------------------------------------------

    processed_groups = []

    for group in choice_groups:

        completed_in_group = [
            course
            for course in group["courses"]
            if course in completed
        ]

        required = group["required"]
        completed_count = len(completed_in_group)

        remaining = max(
            required - completed_count,
            0,
        )

        processed_groups.append(
            {
                "choice_group": group["choice_group"],
                "requirement_type": group["requirement_type"],
                "level": group["level"],
                "required": required,
                "completed": completed_count,
                "remaining": remaining,
                "satisfied": completed_count >= required,
                "completed_courses": completed_in_group,
                "available_courses": group["courses"],
                "notes": group["notes"],
            }
        )

    # ---------------------------------------------------------
    # Determine whether every choice group is satisfied
    # ---------------------------------------------------------

    all_choice_groups_satisfied = all(
        group["satisfied"]
        for group in processed_groups
    )

    # ---------------------------------------------------------
    # Program completion
    #
    # Work terms are excluded from mandatory completion because
    # they apply only to the co-op stream.
    # ---------------------------------------------------------

    eligible = (
        len(missing_individual) == 0
        and all_choice_groups_satisfied
    )

    # ---------------------------------------------------------
    # Calculate completion percentage
    #
    # Individual requirements count as one requirement each.
    #
    # Choice groups count according to the number of courses
    # actually required, not the number of available choices.
    # ---------------------------------------------------------

    total_requirements = (
        len(individual_requirements)
        + sum(
            group["required"]
            for group in processed_groups
        )
    )

    completed_requirement_count = (
        len(completed_individual)
        + sum(
            min(
                group["completed"],
                group["required"],
            )
            for group in processed_groups
        )
    )

    if total_requirements > 0:

        completion_percentage = round(
            (
                completed_requirement_count
                / total_requirements
            )
            * 100,
            1,
        )

    else:

        completion_percentage = 0

    # ---------------------------------------------------------
    # Determine the next level.
    #
    # If the current level is known, the next level is the next
    # numbered program level, unless the student is already at
    # the final level.
    # ---------------------------------------------------------

    next_level = None

    if current_level is not None:

        higher_levels = [
            level
            for level in program_levels
            if level > current_level
        ]

        if higher_levels:
            next_level = min(higher_levels)
        else:
            next_level = current_level

    # ---------------------------------------------------------
    # Build a useful summary for advisor_engine.py
    # ---------------------------------------------------------

    summary = {
        "current_level": current_level,
        "next_level": next_level,
        "final_program_level": final_program_level,
        "evaluation_level": evaluation_level,
        "program_complete": eligible,
        "completion_percentage": completion_percentage,
        "missing_individual_count": len(missing_individual),
        "unsatisfied_choice_group_count": sum(
            1
            for group in processed_groups
            if not group["satisfied"]
        ),
    }

    # ---------------------------------------------------------
    # Return complete result
    # ---------------------------------------------------------

    return {
        "program_id": program_id,

        "eligible": eligible,

        "current_level": current_level,

        "evaluation_level": evaluation_level,

        "final_program_level": final_program_level,

        "next_level": next_level,

        "completion_percentage": completion_percentage,

        "completed_courses": sorted(completed),

        "missing_requirements": missing_individual,

        "choice_groups": processed_groups,

        "optional_requirements": optional_requirements,

        "summary": summary,
    }
