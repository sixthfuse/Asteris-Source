from database import get_connection


def parse_minimum_grade(value):
    if value is None:
        return None

    value = (
        str(value)
        .strip()
        .replace("%", "")
    )

    try:
        return float(value)
    except ValueError:
        return None


def get_required_grade(
    minimum_grade,
    notes,
):
    """
    Get the minimum grade from either the
    minimum_grade column or the notes field.

    Supports values such as:

        50
        "50%"
        "Minimum grade: 50%"
    """

    # ---------------------------------------------------------
    # PREFER DEDICATED COLUMN
    # ---------------------------------------------------------

    required_grade = parse_minimum_grade(
        minimum_grade
    )

    if required_grade is not None:
        return required_grade

    # ---------------------------------------------------------
    # FALL BACK TO NOTES
    # ---------------------------------------------------------

    if notes:

        notes_text = str(notes).strip()

        lower_notes = notes_text.lower()

        if "minimum grade" in lower_notes:

            grade_text = lower_notes.split(
                "minimum grade",
                1,
            )[1]

            if ":" in grade_text:

                grade_text = grade_text.split(
                    ":",
                    1,
                )[1]

            grade_text = grade_text.strip()

            # Extract the first numeric-looking value.
            numeric_text = ""

            for character in grade_text:

                if (
                    character.isdigit()
                    or character == "."
                ):
                    numeric_text += character

                elif numeric_text:
                    break

            if numeric_text:

                try:
                    return float(
                        numeric_text
                    )
                except ValueError:
                    pass

    return None


def check_course_eligibility(
    course_id,
    completed_courses,
    program_id=None,
):
    course_id = course_id.upper()

    completed = {
        item["course_id"].upper():
        item.get("grade")
        for item in completed_courses
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:

            # -------------------------------------------------
            # COURSE
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT
                    course_id,
                    course_name
                FROM courses
                WHERE course_id = %s
                """,
                (course_id,),
            )

            course = cursor.fetchone()

            if course is None:

                return {
                    "course_id": course_id,
                    "eligible": False,
                    "error": "Course not found",
                }

            # -------------------------------------------------
            # PREREQUISITE GROUPS
            # -------------------------------------------------

            cursor.execute(
                """
                SELECT
                    prerequisite_group_id,
                    group_type
                FROM prerequisite_groups
                WHERE course_id = %s
                ORDER BY prerequisite_group_id
                """,
                (course_id,),
            )

            groups = cursor.fetchall()

            missing_requirements = []

            # -------------------------------------------------
            # PROCESS EACH GROUP
            # -------------------------------------------------

            for group_id, group_type in groups:

                cursor.execute(
                    """
                    SELECT
                        prerequisite_course_id,
                        minimum_grade,
                        required_program_id,
                        condition_type,
                        notes
                    FROM prerequisite_conditions
                    WHERE prerequisite_group_id = %s
                    ORDER BY prerequisite_condition_id
                    """,
                    (group_id,),
                )

                conditions = cursor.fetchall()

                # =================================================
                # AND GROUP
                # =================================================

                if group_type.upper() == "AND":

                    for condition in conditions:

                        (
                            prerequisite_course,
                            minimum_grade,
                            required_program_id,
                            condition_type,
                            notes,
                        ) = condition

                        # -----------------------------------------
                        # PROGRAM-SPECIFIC REQUIREMENT
                        # -----------------------------------------

                        if required_program_id:

                            if (
                                program_id
                                != required_program_id
                            ):
                                continue

                        # -----------------------------------------
                        # COURSE REQUIREMENT
                        # -----------------------------------------

                        if condition_type == "COURSE":

                            if not prerequisite_course:
                                continue

                            prerequisite_course = (
                                prerequisite_course.upper()
                            )

                            # -------------------------------------
                            # COURSE NOT COMPLETED
                            # -------------------------------------

                            if (
                                prerequisite_course
                                not in completed
                            ):

                                missing_requirements.append(
                                    {
                                        "type": "COURSE",
                                        "course_id": (
                                            prerequisite_course
                                        ),
                                        "minimum_grade": (
                                            minimum_grade
                                        ),
                                        "notes": notes,
                                    }
                                )

                                continue

                            # -------------------------------------
                            # MINIMUM GRADE
                            # -------------------------------------

                            required_grade = (
                                get_required_grade(
                                    minimum_grade,
                                    notes,
                                )
                            )

                            if required_grade is not None:

                                student_grade = (
                                    completed[
                                        prerequisite_course
                                    ]
                                )

                                # ---------------------------------
                                # GRADE NOT PROVIDED
                                # ---------------------------------

                                if student_grade is None:

                                    missing_requirements.append(
                                        {
                                            "type": "GRADE",
                                            "course_id": (
                                                prerequisite_course
                                            ),
                                            "minimum_grade": (
                                                required_grade
                                            ),
                                            "notes": notes,
                                        }
                                    )

                                # ---------------------------------
                                # GRADE TOO LOW
                                # ---------------------------------

                                elif (
                                    student_grade
                                    < required_grade
                                ):

                                    missing_requirements.append(
                                        {
                                            "type": "GRADE",
                                            "course_id": (
                                                prerequisite_course
                                            ),
                                            "minimum_grade": (
                                                required_grade
                                            ),
                                            "student_grade": (
                                                student_grade
                                            ),
                                            "notes": notes,
                                        }
                                    )

                # =================================================
                # OR GROUP
                # =================================================

                elif group_type.upper() == "OR":

                    group_satisfied = False
                    options = []

                    for condition in conditions:

                        (
                            prerequisite_course,
                            minimum_grade,
                            required_program_id,
                            condition_type,
                            notes,
                        ) = condition

                        # -----------------------------------------
                        # PROGRAM-SPECIFIC REQUIREMENT
                        # -----------------------------------------

                        if required_program_id:

                            if (
                                program_id
                                != required_program_id
                            ):
                                continue

                        if condition_type != "COURSE":
                            continue

                        if not prerequisite_course:
                            continue

                        prerequisite_course = (
                            prerequisite_course.upper()
                        )

                        options.append(
                            prerequisite_course
                        )

                        # -----------------------------------------
                        # COURSE NOT COMPLETED
                        # -----------------------------------------

                        if (
                            prerequisite_course
                            not in completed
                        ):
                            continue

                        # -----------------------------------------
                        # MINIMUM GRADE
                        # -----------------------------------------

                        required_grade = (
                            get_required_grade(
                                minimum_grade,
                                notes,
                            )
                        )

                        # No minimum grade means completion
                        # alone satisfies this option.
                        if required_grade is None:

                            group_satisfied = True
                            break

                        student_grade = (
                            completed[
                                prerequisite_course
                            ]
                        )

                        if (
                            student_grade is not None
                            and student_grade
                            >= required_grade
                        ):

                            group_satisfied = True
                            break

                    # -----------------------------------------
                    # NO OR OPTION SATISFIED
                    # -----------------------------------------

                    if not group_satisfied:

                        missing_requirements.append(
                            {
                                "type": "OR",
                                "options": options,
                            }
                        )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    return {
        "course_id": course_id,
        "course_name": course[1],
        "eligible": (
            len(missing_requirements) == 0
        ),
        "completed_courses": completed_courses,
        "missing_requirements": (
            missing_requirements
        ),
    }