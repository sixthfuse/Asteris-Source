from advisor_eligibility import find_eligible_courses


def get_next_courses(
    program_id,
    completed_courses,
):
    program_id = program_id.upper()

    result = find_eligible_courses(
        program_id=program_id,
        completed_courses=completed_courses,
    )

    if "error" in result:
        return result

    current_level = result["current_level"]
    eligible_courses = result["eligible_courses"]

    current_level_courses = []
    other_eligible_courses = []

    for course in eligible_courses:

        if course["level"] == current_level:
            current_level_courses.append(course)
        else:
            other_eligible_courses.append(course)

    core_courses = [
        course
        for course in current_level_courses
        if course["course_type"] == "Core"
    ]

    elective_courses = [
        course
        for course in current_level_courses
        if "Elective" in course["course_type"]
    ]

    work_terms = [
        course
        for course in current_level_courses
        if course["course_type"] == "Work Term"
    ]

    return {
        "program_id": program_id,
        "current_level": current_level,
        "next_courses": current_level_courses,
        "core_courses": core_courses,
        "elective_courses": elective_courses,
        "work_terms": work_terms,
        "other_eligible_courses": other_eligible_courses,
        "next_course_count": len(current_level_courses),
    }