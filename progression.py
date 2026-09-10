from database import get_connection


def check_progression(
    program_id,
    from_level,
    completed_courses,
    gpa=None,
    work_hours=0,
    diploma_completed=False,
):
    program_id = program_id.upper()

    completed = {
        item["course_id"].upper()
        for item in completed_courses
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    progression_requirement_id,
                    from_level,
                    to_level,
                    requirement_type,
                    requirement_value,
                    description
                FROM progression_requirements
                WHERE program_id = %s
                  AND from_level = %s
                ORDER BY progression_requirement_id
                """,
                (program_id, from_level),
            )

            requirements = cursor.fetchall()

    if not requirements:
        return {
            "program_id": program_id,
            "from_level": from_level,
            "eligible": False,
            "error": "Progression requirements not found",
        }

    missing_requirements = []
    satisfied_requirements = []

    to_level = requirements[0][2]

    for requirement in requirements:

        (
            requirement_id,
            requirement_from_level,
            requirement_to_level,
            requirement_type,
            requirement_value,
            description,
        ) = requirement

        satisfied = False
        details = None

        # -----------------------------------------------------
        # COURSE COMPLETION
        # -----------------------------------------------------

        if requirement_type == "COURSE_COMPLETION":

            course_id = str(
                requirement_value
            ).upper()

            satisfied = course_id in completed

            details = {
                "course_id": course_id,
                "completed": satisfied,
            }

        # -----------------------------------------------------
        # DIPLOMA COMPLETION
        # -----------------------------------------------------

        elif requirement_type == "DIPLOMA_COMPLETION":

            satisfied = diploma_completed

            details = {
                "diploma_completed": diploma_completed,
            }

        # -----------------------------------------------------
        # WORK EXPERIENCE
        # -----------------------------------------------------

        elif requirement_type == "WORK_EXPERIENCE":

            value = str(
                requirement_value
            ).lower()

            value = (
                value
                .replace("hours", "")
                .strip()
            )

            try:
                required_hours = float(value)
            except ValueError:
                required_hours = None

            if required_hours is not None:

                satisfied = (
                    work_hours >= required_hours
                )

                details = {
                    "required_hours": required_hours,
                    "completed_hours": work_hours,
                }

        # -----------------------------------------------------
        # GPA
        # -----------------------------------------------------

        elif requirement_type == "GPA":

            value = str(
                requirement_value
            ).replace("%", "").strip()

            try:
                required_gpa = float(value)
            except ValueError:
                required_gpa = None

            if (
                required_gpa is not None
                and gpa is not None
            ):

                satisfied = (
                    gpa >= required_gpa
                )

                details = {
                    "required_gpa": required_gpa,
                    "student_gpa": gpa,
                }

        # -----------------------------------------------------
        # SATISFIED / MISSING
        # -----------------------------------------------------

        if satisfied:

            satisfied_requirements.append(
                {
                    "requirement_id": requirement_id,
                    "requirement_type": requirement_type,
                    "requirement_value": requirement_value,
                    "description": description,
                    "details": details,
                }
            )

        else:

            missing_requirements.append(
                {
                    "requirement_id": requirement_id,
                    "requirement_type": requirement_type,
                    "requirement_value": requirement_value,
                    "description": description,
                    "details": details,
                }
            )

    eligible = (
        len(missing_requirements) == 0
    )

    return {
        "program_id": program_id,
        "from_level": from_level,
        "to_level": to_level,
        "eligible": eligible,
        "satisfied_requirements": (
            satisfied_requirements
        ),
        "missing_requirements": (
            missing_requirements
        ),
    }