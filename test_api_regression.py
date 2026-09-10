"""Regression tests for the running Asteris advisor API.

Start Uvicorn in one terminal before running this file. These tests only
POST student scenarios to the local API; they do not change database data.
"""

import json
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/advisor/engine"

BENG_INCOMPLETE_IDS = """
CIVL1012 CIVL1020 CIVL1024 CIVL1060 COMM1142 MATH1422 PHYS1192 SURV1130
CIVL2020 CIVL2024 CIVL2025 CIVL2026 COMM2242 MATH2422 MATH2423 PHYS2192 SURV2230
CIVL3012 CIVL3020 CIVL3033 CIVL3041 CIVL3050 CIVL3052 CIVL3074 COMM3342 MATH3423
CIVL4033 CIVL4041 CIVL4052 CIVL4074 CIVL4090 COMM4442 CHEM6020 MATH6010
CIVL7012 CIVL7021 CIVL7060 CIVL7070 ELEX7355 MATH7010
CIVL7001 CIVL7011 CIVL7020 CIVL7022 CIVL7040 CIVL7092 LIBS7005
CIVL7023 CIVL7033 CIVL7062 CIVL7072 CIVL7089 LIBS7025 MINE7041 CIVL7063
CIVL7030 CIVL7050 CIVL7090 CIVL7042 CIVL7061
""".split()

DIPLOMA_COMPLETE_IDS = """
CIVL1012 CIVL1020 CIVL1024 CIVL1060 COMM1142 MATH1422 PHYS1192 SURV1130
CIVL2020 CIVL2024 CIVL2025 CIVL2026 COMM2242 MATH2422 MATH2423 PHYS2192 SURV2230
CIVL3012 CIVL3020 CIVL3033 CIVL3041 CIVL3050 CIVL3052 CIVL3074 COMM3342 MATH3423
CIVL4033 CIVL4041 CIVL4052 CIVL4074 CIVL4090 COMM4442 CHEM6020 MATH6010
""".split()

CIRCULAR_ECONOMY_IDS = ["XCIR7510", "XCIR7520", "XCIR7530"]


def completed_courses(course_ids, minimum_grade_overrides=None):
    """Build the API's completed-course format from course IDs."""

    minimum_grade_overrides = minimum_grade_overrides or {}

    return [
        {
            "course_id": course_id,
            "grade": minimum_grade_overrides.get(course_id, 75),
        }
        for course_id in course_ids
    ]


class AdvisorApiRegressionTests(unittest.TestCase):
    """End-to-end checks for the three finalized program records."""

    def post_engine(self, program_id, courses):
        payload = json.dumps(
            {
                "program_id": program_id,
                "completed_courses": courses,
                "gpa": 75,
                "work_hours": 300,
                "diploma_completed": True,
            }
        ).encode("utf-8")

        request = Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 200)
                return json.loads(response.read().decode("utf-8"))
        except URLError as error:
            self.fail(
                "Cannot reach the Asteris API. Keep Uvicorn running "
                f"at {API_URL}. Details: {error}"
            )

    def test_beng_incomplete_when_opmt7031_is_missing(self):
        result = self.post_engine(
            "8660BENG",
            completed_courses(
                BENG_INCOMPLETE_IDS,
                {"CIVL7011": 54},
            ),
        )

        self.assertFalse(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 98.4)
        self.assertEqual(result["evaluation_level"], 8)
        self.assertEqual(result["progression_status"], "FINAL_LEVEL_ENTRY")

        missing_ids = {
            item["course_id"]
            for item in result["program_progress"]["missing_requirements"]
        }
        self.assertEqual(missing_ids, {"OPMT7031"})

        blocked_ids = {
            item["course_id"]
            for item in result["blocked_courses"]
        }
        self.assertIn("OPMT7031", blocked_ids)

    def test_beng_complete_after_opmt7031_and_passing_prerequisite(self):
        result = self.post_engine(
            "8660BENG",
            completed_courses(
                BENG_INCOMPLETE_IDS + ["OPMT7031"],
                {"CIVL7011": 55},
            ),
        )

        self.assertTrue(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 100.0)
        self.assertEqual(result["current_level"], 8)
        self.assertEqual(
            result["program_progress"]["missing_requirements"],
            [],
        )

    def test_diploma_new_student_starts_at_level_one(self):
        result = self.post_engine("5410DIPLT", [])

        self.assertFalse(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 0)
        self.assertEqual(result["current_level"], 0)
        self.assertEqual(result["evaluation_level"], 1)
        self.assertEqual(result["next_level"], 1)
        self.assertEqual(result["progression_status"], "INITIAL_LEVEL")

        next_course_ids = {
            course["course_id"]
            for course in result["next_courses"]
        }
        self.assertEqual(
            next_course_ids,
            {
                "CIVL1012",
                "CIVL1020",
                "CIVL1024",
                "CIVL1060",
                "COMM1142",
                "MATH1422",
                "PHYS1192",
                "SURV1130",
            },
        )

    def test_diploma_complete_with_two_level_four_electives(self):
        result = self.post_engine(
            "5410DIPLT",
            completed_courses(DIPLOMA_COMPLETE_IDS),
        )

        self.assertTrue(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 100.0)
        self.assertEqual(result["current_level"], 4)
        self.assertEqual(
            result["program_progress"]["missing_requirements"],
            [],
        )

    def test_circular_economy_new_student_sees_all_three_courses(self):
        result = self.post_engine("0816CM", [])

        self.assertFalse(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 0)
        self.assertEqual(result["evaluation_level"], 1)
        self.assertEqual(result["progression_status"], "INITIAL_LEVEL")

        next_course_ids = {
            course["course_id"]
            for course in result["next_courses"]
        }
        self.assertEqual(next_course_ids, set(CIRCULAR_ECONOMY_IDS))

    def test_circular_economy_complete_after_all_three_courses(self):
        result = self.post_engine(
            "0816CM",
            completed_courses(CIRCULAR_ECONOMY_IDS),
        )

        self.assertTrue(result["program_complete"])
        self.assertEqual(result["completion_percentage"], 100.0)
        self.assertEqual(result["current_level"], 1)
        self.assertEqual(
            result["program_progress"]["missing_requirements"],
            [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
