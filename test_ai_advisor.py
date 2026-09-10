"""Unit tests for Phase 5 AI routing; no live API or database required."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app

from ai_advisor import (
    AdvisorIntent,
    answer_student_question,
    classify_intent,
    execute_tool,
    extract_conversation_completed_courses,
    find_programs,
    get_program_electives,
    get_program_progression_requirements,
    get_program_details,
    get_student_level_requirements,
    get_student_level_readiness,
    hide_internal_program_ids,
    current_message_language,
    is_global_catalog_question,
    is_out_of_scope_question,
    normalize_course_id,
    resolve_academic_context,
    requested_target_level,
)


class FakeResponses:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


class AiAdvisorTests(unittest.TestCase):
    def _advisor_client(self, tool_name, arguments, final_answer="Verified catalog results."):
        call = SimpleNamespace(
            type="function_call", name=tool_name,
            arguments=json.dumps(arguments), call_id=f"{tool_name}-call",
        )
        return FakeClient([
            SimpleNamespace(id="tool-request", output=[call], output_text=""),
            SimpleNamespace(id="final-answer", output=[], output_text=final_answer),
        ])

    @patch("ai_advisor.OpenAI")
    def test_advisor_returns_every_stored_program_url(self, openai_client):
        client = self._advisor_client("search_programs", {"query": "all programs"})
        openai_client.return_value = client
        response = TestClient(app).post(
            "/advisor", json={"question": "Give me the web links for all 3 programs."}
        )
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        payload = json.loads(client.responses.requests[1]["input"][0]["output"])
        self.assertEqual(payload["active_program_count"], 3)
        self.assertTrue(all(program["source_url"] for program in payload["programs"]))
        for program in payload["programs"]:
            self.assertIn(program["source_url"], answer)

    @patch("ai_advisor.OpenAI")
    def test_advisor_explicit_program_id_request_returns_all_real_ids(self, openai_client):
        client = self._advisor_client("search_programs", {"query": "all programs"})
        openai_client.return_value = client
        response = TestClient(app).post(
            "/advisor", json={"question": "Give me the IDs for all 3 programs."}
        )
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        for program_id in ("0816CM", "8660BENG", "5410DIPLT"):
            self.assertIn(program_id, answer)
        self.assertNotIn("the program", answer.lower())

    def test_course_id_request_is_not_misclassified_as_program_id_request(self):
        from ai_advisor import explicitly_requests_program_ids

        self.assertFalse(explicitly_requests_program_ids(
            "Give me all course IDs and credits for the Civil Engineering bachelor program."
        ))
        self.assertTrue(explicitly_requests_program_ids(
            "Give me the IDs for the 3 programs."
        ))

    @patch("ai_advisor.OpenAI")
    def test_advisor_normal_program_answer_can_still_hide_program_id(self, openai_client):
        client = self._advisor_client(
            "get_program_details", {"program_id": "8660BENG"},
            "Civil Engineering Bachelor of Engineering uses program 8660BENG.",
        )
        openai_client.return_value = client
        response = TestClient(app).post(
            "/advisor", json={"question": "Tell me about the Civil Engineering bachelor program."}
        )
        answer = response.json()["answer"]
        self.assertNotIn("8660BENG", answer)
        self.assertIn("the program", answer)

    @patch("ai_advisor.OpenAI")
    def test_advisor_course_name_is_always_paired_with_course_code(self, openai_client):
        client = self._advisor_client(
            "get_course_details", {"course_id": "CIVL2025"},
            "Applied Hydraulics covers fluid mechanics applications.",
        )
        openai_client.return_value = client
        response = TestClient(app).post(
            "/advisor", json={"question": "Tell me about Applied Hydraulics."}
        )
        self.assertIn("CIVL2025 Applied Hydraulics", response.json()["answer"])

    @patch("ai_advisor.OpenAI")
    def test_advisor_beng_course_credit_list_uses_enriched_database_values(self, openai_client):
        final_answer = (
            "The list includes CIVL2020 Mechanics of Materials 1 — 6.50 credits, "
            "and CIVL2025 Applied Hydraulics — 5.50 credits."
        )
        client = self._advisor_client(
            "get_program_details", {"program_id": "8660BENG"}, final_answer,
        )
        openai_client.return_value = client
        response = TestClient(app).post("/advisor", json={
            "question": "Give me all course IDs and credits for the Civil Engineering bachelor program."
        })
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        payload = json.loads(client.responses.requests[1]["input"][0]["output"])["program"]
        self.assertEqual(len(payload["courses"]), 73)
        courses = {course["course_id"]: course for course in payload["courses"]}
        self.assertEqual(courses["CIVL2025"]["credits"], "5.50")
        self.assertEqual(courses["CIVL2020"]["credits"], "6.50")
        for field in ("course_name", "credits", "source_url", "level", "term",
                      "course_type", "required", "notes"):
            self.assertIn(field, courses["CIVL2025"])
        self.assertIn("CIVL2025", answer)
        self.assertIn("5.50", answer)
        self.assertNotIn("credits unavailable", answer.lower())
        self.assertNotIn("not available", answer.lower())

    def test_spanish_and_english_missing_level_questions_route_to_progress(self):
        questions = (
            "What do I still need for Level 3?",
            "¿Qué me falta para el nivel 3?",
            "Que necesito para el nivel tres?",
        )
        for question in questions:
            self.assertEqual(classify_intent(question), AdvisorIntent.PROGRAM_PROGRESS)
            self.assertEqual(requested_target_level(question), 3)
        for question in (
            "What do I need to finish Level 6?",
            "What remains to finish level six?",
            "Which requirements are left for Level 6?",
        ):
            self.assertEqual(classify_intent(question), AdvisorIntent.PROGRAM_PROGRESS)

    def test_noisy_spanish_completed_list_is_temporary_conversation_context(self):
        text = (
            "Terminé estas clases (copiado del portal): CIVL-2020 ✓ | CIVL 2024; "
            "CIVL:2025, CIVL_2026 / COMM 2242 y MATH-2422. INVALID 9999"
        )
        courses, added = extract_conversation_completed_courses(text, [], [])
        self.assertEqual(
            set(added),
            {"CIVL2020", "CIVL2024", "CIVL2025", "CIVL2026", "COMM2242", "MATH2422"},
        )
        self.assertEqual({course["course_id"] for course in courses}, set(added))

    def test_have_taken_course_list_is_temporary_conversation_context(self):
        text = "I have taken these courses: CIVL 7001, CIVL 7011, CIVL 7020, CIVL 7022, CIVL 7040"
        courses, added = extract_conversation_completed_courses(text, [], [])
        self.assertEqual(len(courses), 5)
        self.assertEqual(len(added), 5)

    def test_course_in_incomplete_question_is_not_completed(self):
        text = (
            "I haven't completed my 1st year of civil engineering program, "
            "can I take this course? COMM 3342"
        )
        courses, added = extract_conversation_completed_courses(text, [], [])
        self.assertEqual(courses, [])
        self.assertEqual(added, [])
        result = execute_tool(
            "check_course_eligibility",
            {"course_id": "COMM 3342"},
            {"program_id": "8660BENG", "completed_courses": courses},
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["course_id"], "COMM3342")
        self.assertNotIn("completed_courses", result)
        self.assertEqual(result["missing_requirements"][0]["type"], "PROGRAM_LEVEL")

    def test_follow_up_uses_completion_from_temporary_conversation_context(self):
        courses, added = extract_conversation_completed_courses(
            "Can I take the next course?",
            [{"role": "user", "content": "I completed CIVL 7001 last term."}],
            [],
        )
        self.assertEqual(added, ["CIVL7001"])
        self.assertEqual(courses, [{"course_id": "CIVL7001", "grade": None}])

    def test_student_answer_never_mentions_academic_profile(self):
        client = FakeClient([
            SimpleNamespace(
                id="r1", output=[],
                output_text="Your academic profile lists COMM3342 as completed.",
            )
        ])
        result = answer_student_question("Can I take COMM 3342?", [], client=client)
        self.assertNotIn("academic profile", result["answer"].lower())
        self.assertIn("courses you told me about", result["answer"].lower())

    def test_app_contains_chat_only_and_no_profile_controls(self):
        response = TestClient(app).get("/app")
        self.assertEqual(response.status_code, 200)
        html = response.text.lower()
        self.assertIn('id="chat-form"', html)
        for forbidden in (
            "academic profile", 'id="program"', 'id="courses"', 'id="gpa"',
            'id="hours"', 'id="diploma"',
        ):
            self.assertNotIn(forbidden, html)

    def test_level_three_readiness_returns_exact_missing_level_two_courses(self):
        completed = [
            {"course_id": course_id, "grade": None}
            for course_id in (
                "CIVL2020", "CIVL2024", "CIVL2025", "CIVL2026", "COMM2242", "MATH2422"
            )
        ]
        result = get_student_level_readiness("8660BENG", 3, completed)
        self.assertFalse(result["ready"])
        self.assertEqual(
            [course["course_id"] for course in result["missing_courses"]],
            ["MATH2423", "PHYS2192", "SURV2230"],
        )

    def test_level_ampersand_gets_specific_clarification(self):
        result = answer_student_question(
            "I have 600 hours of work experience, can I move on to level &?", []
        )
        self.assertEqual(result["answer"], "Did you mean Level 7?")
        self.assertEqual(result["tools_used"], [])

    def test_program_website_follow_up_exposes_program_detail_tool(self):
        client = FakeClient([SimpleNamespace(id="r1", output=[], output_text="Program link")])
        answer_student_question(
            "Do you have a website for it?",
            [],
            conversation=[{
                "role": "assistant",
                "content": "Applied Circular Economy: Zero Waste Buildings is a microcredential.",
            }],
            client=client,
        )
        tool_names = {tool["name"] for tool in client.responses.requests[0]["tools"]}
        self.assertIn("get_program_details", tool_names)

    def test_paraphrases_route_to_structural_advisor_tools(self):
        for text in ("Which electives can I choose?", "Show me the choice groups"):
            self.assertEqual(classify_intent(text), AdvisorIntent.ELECTIVES)
        for text in ("How much practical work do I need?", "What lets me move to level 7?"):
            self.assertEqual(classify_intent(text), AdvisorIntent.PROGRESSION)
        for text in ("Do you have microcredentials?", "What programs are available?"):
            self.assertEqual(classify_intent(text), AdvisorIntent.PROGRAM_CATEGORY)

    def test_category_discovery_uses_program_credentials(self):
        programs = find_programs("Do you offer any microcredentials?")
        self.assertTrue(programs)
        self.assertTrue(all("microcredential" in p["credential"].lower() for p in programs))
        self.assertGreaterEqual(len(find_programs("What programs are available?")), 3)

    def test_global_catalog_queries_use_all_active_program_records(self):
        questions = (
            "how many programs do you offer?",
            "give me a list of all credentials",
            "give me a list of all programs you offer",
        )
        for question in questions:
            self.assertTrue(is_global_catalog_question(question))
            self.assertEqual(classify_intent(question), AdvisorIntent.PROGRAM_CATEGORY)
            self.assertEqual(len(find_programs(question)), 3)
        self.assertEqual(
            {program["credential"] for program in find_programs(questions[1])},
            {"Bachelor's Degree", "Diploma", "Microcredential"},
        )

    def test_spanish_global_catalog_paraphrases_use_full_catalog(self):
        questions = (
            "¿Cuántos programas ofrecen en BCIT?",
            "¿Qué programas ofrecen?",
            "¿Cuántas carreras ofrecen en BCIT?",
            "lista de programas",
            "Dame una lista de todas las carreras",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_global_catalog_question(question))
                self.assertEqual(classify_intent(question), AdvisorIntent.PROGRAM_CATEGORY)
                self.assertEqual(len(find_programs(question)), 3)

    def test_current_turn_language_overrides_conversation_language(self):
        cases = (
            (
                "How many programs are offered at BCIT?",
                [{"role": "user", "content": "¿Qué programas ofrecen?"}],
                "English",
            ),
            (
                "¿Cuántos programas ofrecen en BCIT?",
                [{"role": "user", "content": "What programs are available?"}],
                "Spanish",
            ),
        )
        for question, conversation, expected_language in cases:
            with self.subTest(question=question):
                client = self._advisor_client("search_programs", {"query": question})
                answer_student_question(question, [], conversation=conversation, client=client)
                first_request = client.responses.requests[0]
                self.assertEqual(current_message_language(question), expected_language)
                self.assertIn(
                    f"Required response language from current turn: {expected_language}",
                    first_request["input"],
                )
                self.assertIn(
                    f"language of the current student question: {expected_language}",
                    first_request["instructions"],
                )

    @patch("ai_advisor.OpenAI")
    def test_general_knowledge_questions_are_declined_without_model_or_tools(self, openai_client):
        for question in ("Do you know who Shakira is?", "Who won the World Cup?"):
            with self.subTest(question=question):
                result = answer_student_question(question, [])
                self.assertEqual(result["intent"], AdvisorIntent.OUT_OF_SCOPE.value)
                self.assertEqual(result["tools_used"], [])
                self.assertIn("BCIT", result["answer"])
        openai_client.assert_not_called()

    def test_bcit_career_question_remains_in_scope(self):
        question = "I'm interested in becoming a structural engineer; what BCIT program should I look at?"
        self.assertFalse(is_out_of_scope_question(question))
        self.assertNotEqual(classify_intent(question), AdvisorIntent.OUT_OF_SCOPE)

    def test_global_catalog_turn_overrides_microcredential_context(self):
        conversation = [
            {"role": "user", "content": "Tell me about the microcredential."},
            {"role": "assistant", "content": "Applied Circular Economy: Zero Waste Buildings."},
        ]
        for question in (
            "how many programs do you offer?",
            "give me a list of all credentials",
            "give me a list of all programs you offer",
        ):
            context = resolve_academic_context(question, conversation, "0816CM")
            self.assertTrue(context["global_catalog"])
            self.assertIsNone(context["program_id"])
            self.assertIsNone(context["program_name"])

    def test_global_catalog_tool_summary_is_derived_from_program_rows(self):
        result = execute_tool(
            "search_programs", {"query": "microcredential"}, {"global_catalog": True}
        )
        self.assertEqual(result["active_program_count"], 3)
        self.assertEqual(len(result["programs"]), 3)
        self.assertEqual(
            set(result["distinct_credentials"]),
            {"Bachelor's Degree", "Diploma", "Microcredential"},
        )

    def test_student_ui_builds_safe_clickable_links_without_inner_html(self):
        with open("static/app.js", encoding="utf-8") as source:
            javascript = source.read()
        self.assertIn('document.createElement("a")', javascript)
        self.assertIn("anchor.textContent =", javascript)
        self.assertIn('anchor.rel = "noopener noreferrer"', javascript)
        self.assertNotIn("innerHTML", javascript)

    def test_internal_program_ids_are_hidden_from_students(self):
        self.assertEqual(
            hide_internal_program_ids("You are enrolled in 8660BENG."),
            "You are enrolled in the program.",
        )
        source_url = (
            "https://www.bcit.ca/programs/applied-circular-economy-zero-waste-"
            "buildings-microcredential-part-time-0816cm/"
        )
        self.assertEqual(hide_internal_program_ids(source_url), source_url)

    def test_elective_groups_exclude_work_terms(self):
        groups = get_program_electives("8660BENG")
        self.assertTrue(groups)
        serialized = json.dumps(groups).lower()
        self.assertNotIn("work term", serialized)
        self.assertNotIn("civl5990", serialized)
        self.assertTrue(all(group["choose"] for group in groups))

    def test_progression_returns_practical_work_rules_and_sources(self):
        requirements = get_program_progression_requirements("8660BENG")
        work = [r for r in requirements if r["type"] == "WORK_EXPERIENCE"]
        self.assertEqual([r["value"] for r in work], ["300 hours", "700 hours"])
        self.assertTrue(all(r["source_url"] for r in work))

    def test_program_context_persists_across_natural_follow_up(self):
        context = resolve_academic_context(
            "What electives can I choose?",
            [{"role": "user", "content": "Tell me about the Civil Engineering bachelor."},
             {"role": "assistant", "content": "The Civil Engineering Bachelor of Engineering is active."}],
            None,
        )
        self.assertEqual(context["program_name"], "Civil Engineering Bachelor of Engineering")

    def test_level_four_elective_follow_up_inherits_structured_context(self):
        context = resolve_academic_context(
            "what are the options of the electives?",
            [{"role": "user", "content": "How many electives do I need to choose from the level 4, civil engineering program?"},
             {"role": "assistant", "content": "At Level 4, choose 2 electives."}],
            None,
        )
        self.assertEqual(context["program_name"], "Civil Engineering Bachelor of Engineering")
        self.assertEqual(context["level"], 4)
        self.assertEqual(context["requirement_type"], "electives")
        groups = get_program_electives(context["program_id"], context["level"])
        self.assertEqual(groups[0]["choose"], 2)
        self.assertEqual(
            [course["course_id"] for course in groups[0]["courses"]],
            ["CHEM6020", "CIVL4024", "CIVL4053", "MATH6010"],
        )
        self.assertNotIn("CIVL5990", json.dumps(groups))

    def test_level_six_completion_aggregates_core_choice_and_optional(self):
        completed = [{"course_id": course_id} for course_id in (
            "CIVL7001", "CIVL7011", "CIVL7020", "CIVL7022", "CIVL7040"
        )]
        result = get_student_level_requirements("8660BENG", 6, completed)
        self.assertEqual([item["course_id"] for item in result["missing_required_courses"]],
                         ["CIVL7092"])
        self.assertEqual(result["unsatisfied_choice_groups"][0]["required"], 1)
        self.assertEqual(
            [item["course_id"] for item in result["unsatisfied_choice_groups"][0]["available_options"]],
            ["LIBS7005", "LIBS7007"],
        )
        self.assertEqual([item["course_id"] for item in result["optional_requirements"]],
                         ["CIVL6990"])

    def test_microcredential_details_aggregate_courses_and_source(self):
        programs = find_programs("Applied Circular Economy Zero Waste Buildings")
        details = get_program_details(programs[0]["program_id"])
        self.assertTrue(details["courses"])
        self.assertTrue(details["source_url"])

    def test_microcredential_detail_questions_resolve_and_force_aggregate_tool(self):
        questions = (
            "can you give me more information on the microcredential program you offer?",
            "Can you provide course names and IDs for the microcredential program you offer?",
        )
        for question in questions:
            call = SimpleNamespace(
                type="function_call", name="get_program_details",
                arguments='{"program_id":"0816CM"}', call_id="program-details",
            )
            final_answer = (
                "Applied Circular Economy: Zero Waste Buildings includes "
                "XCIR7510 Deconstruction Management, XCIR7520 Design for Disassembly, "
                "and XCIR7530 Construction Material Flows. "
                "https://www.bcit.ca/programs/applied-circular-economy-zero-waste-buildings-microcredential-part-time-0816cm/"
            )
            client = FakeClient([
                SimpleNamespace(id="r1", output=[call], output_text=""),
                SimpleNamespace(id="r2", output=[], output_text=final_answer),
            ])

            result = answer_student_question(question, [], client=client)

            self.assertEqual(result["intent"], "PROGRAM_INFO")
            self.assertEqual(result["resolved_program"], "Applied Circular Economy: Zero Waste Buildings")
            self.assertEqual(result["tools_used"], ["get_program_details"])
            self.assertEqual(
                client.responses.requests[0]["tool_choice"],
                {"type": "function", "name": "get_program_details"},
            )
            tool_payload = json.loads(client.responses.requests[1]["input"][0]["output"])["program"]
            self.assertEqual(
                [(course["course_id"], course["course_name"]) for course in tool_payload["courses"]],
                [
                    ("XCIR7510", "Deconstruction Management"),
                    ("XCIR7520", "Design for Disassembly"),
                    ("XCIR7530", "Construction Material Flows"),
                ],
            )
            self.assertTrue(tool_payload["source_url"])
            for expected in (
                "XCIR7510", "Deconstruction Management", "XCIR7520",
                "Design for Disassembly", "XCIR7530", "Construction Material Flows",
                tool_payload["source_url"],
            ):
                self.assertIn(expected, result["answer"])

    def test_microcredential_follow_up_inherits_resolved_program(self):
        conversation = [
            {"role": "user", "content": "can you give me more information on the microcredential program you offer?"},
            {"role": "assistant", "content": "Applied Circular Economy: Zero Waste Buildings is the microcredential."},
        ]
        context = resolve_academic_context(
            "Can you provide course names and IDs for the microcredential program you offer?",
            conversation,
            None,
        )
        self.assertEqual(context["program_id"], "0816CM")
        self.assertEqual(context["program_name"], "Applied Circular Economy: Zero Waste Buildings")

    @patch("ai_advisor.OpenAI")
    def test_ui_advisor_endpoint_returns_microcredential_details_across_follow_up(
        self, openai_client
    ):
        source_url = (
            "https://www.bcit.ca/programs/applied-circular-economy-zero-waste-"
            "buildings-microcredential-part-time-0816cm/"
        )
        first_answer = (
            "Applied Circular Economy: Zero Waste Buildings includes XCIR7510 "
            "Deconstruction Management, XCIR7520 Design for Disassembly, and "
            f"XCIR7530 Construction Material Flows. {source_url}"
        )
        second_answer = (
            "The courses are XCIR7510 Deconstruction Management, XCIR7520 Design "
            "for Disassembly, and XCIR7530 Construction Material Flows."
        )

        def detail_client(answer, call_id):
            call = SimpleNamespace(
                type="function_call", name="get_program_details",
                arguments='{"program_id":"0816CM"}', call_id=call_id,
            )
            return FakeClient([
                SimpleNamespace(id=f"{call_id}-1", output=[call], output_text=""),
                SimpleNamespace(id=f"{call_id}-2", output=[], output_text=answer),
            ])

        openai_client.side_effect = [
            detail_client(first_answer, "details-first"),
            detail_client(second_answer, "details-follow-up"),
        ]
        http = TestClient(app)
        first_question = "can you give me more information on the microcredential program you offer?"
        first = http.post("/advisor", json={"question": first_question})
        self.assertEqual(first.status_code, 200)
        first_result = first.json()
        follow_up = http.post("/advisor", json={
            "question": "Can you provide course names and IDs for the microcredential program you offer?",
            "conversation": [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first_result["answer"]},
            ],
        })
        self.assertEqual(follow_up.status_code, 200)
        follow_up_result = follow_up.json()

        for result in (first_result, follow_up_result):
            self.assertEqual(result["tools_used"], ["get_program_details"])
            for expected in (
                "XCIR7510", "Deconstruction Management", "XCIR7520",
                "Design for Disassembly", "XCIR7530", "Construction Material Flows",
            ):
                self.assertIn(expected, result["answer"])
        self.assertIn(source_url, first_result["answer"])

    def test_student_language_routes_to_specific_modes(self):
        self.assertEqual(classify_intent("Hydraulics"), AdvisorIntent.COURSE_INFO)
        self.assertEqual(
            classify_intent("Tell me about the Civil Engineering diploma program"),
            AdvisorIntent.PROGRAM_INFO,
        )
        self.assertEqual(
            classify_intent("What are the prerequisites for Hydraulics?"),
            AdvisorIntent.PREREQUISITES,
        )
        self.assertEqual(
            classify_intent("Can I take Hydraulics next?"),
            AdvisorIntent.ELIGIBILITY_NEXT_COURSES,
        )
        self.assertEqual(
            classify_intent("How much of my program is complete?"),
            AdvisorIntent.PROGRAM_PROGRESS,
        )
        self.assertEqual(
            classify_intent(
                "I just finished all my level two courses, what percentage of the "
                "program am I in, for the whole bachelor degree program?"
            ),
            AdvisorIntent.PROGRAM_PROGRESS,
        )

    def test_hydraulics_info_cannot_invoke_student_progress_tools(self):
        client = FakeClient([SimpleNamespace(id="r1", output=[], output_text="Course info")])

        result = answer_student_question("Hydraulics", [], client=client)

        request = client.responses.requests[0]
        tool_names = {tool["name"] for tool in request["tools"]}
        self.assertEqual(tool_names, {"search_courses", "get_course_details"})
        self.assertNotIn("Completed course count", request["input"])
        self.assertEqual(result["intent"], "COURSE_INFO")

    def test_program_name_query_cannot_invoke_progress_tools_or_request_id(self):
        client = FakeClient([SimpleNamespace(id="r1", output=[], output_text="Program info")])

        result = answer_student_question(
            "Tell me about the Civil Engineering diploma program", [], client=client
        )

        request = client.responses.requests[0]
        tool_names = {tool["name"] for tool in request["tools"]}
        self.assertEqual(
            tool_names,
            {"search_programs", "get_program_details", "get_program_level_courses"},
        )
        self.assertNotIn("Completed course count", request["input"])
        self.assertIn("never ask a student for a program ID", request["instructions"])
        self.assertEqual(result["intent"], "PROGRAM_INFO")

    @patch("ai_advisor.infer_completed_levels")
    @patch("ai_advisor.resolve_civil_engineering_context")
    def test_whole_bachelor_progress_resolves_pathway_and_uses_level_claim(
        self, resolve_context, infer_levels
    ):
        resolve_context.return_value = {
            "program_id": "8660BENG",
            "pathway": {"name": "Civil Engineering"},
        }
        inferred = [{"course_id": "CIVL1012", "grade": None}]
        infer_levels.return_value = (inferred, [1, 2])
        call = SimpleNamespace(
            type="function_call", name="get_student_program_advice",
            arguments="{}", call_id="progress-1",
        )
        client = FakeClient([
            SimpleNamespace(id="r1", output=[call], output_text=""),
            SimpleNamespace(id="r2", output=[], output_text="You have completed 25%."),
        ])

        with patch("ai_advisor.execute_tool", return_value={"completion_percentage": 25}) as tool:
            result = answer_student_question(
                "I just finished all my level two courses, what percentage of the "
                "program am I in, for the whole bachelor degree program?",
                [],
                conversation=[
                    {"role": "user", "content": "Tell me about Civil Engineering."}
                ],
                client=client,
            )

        self.assertEqual(result["intent"], "PROGRAM_PROGRESS")
        self.assertEqual(result["tools_used"], ["get_student_program_advice"])
        context = tool.call_args.args[2]
        self.assertEqual(context["program_id"], "8660BENG")
        self.assertEqual(context["completed_courses"], inferred)
        self.assertIn("through Level 2", client.responses.requests[0]["input"])
        self.assertNotIn("official outline", result["answer"].lower())

    @patch("ai_advisor.resolve_civil_engineering_context")
    def test_shared_level_three_query_lists_courses_without_credential_question(
        self, resolve_context
    ):
        resolve_context.return_value = {
            "program_id": "8660BENG",
            "pathway": {"name": "Civil Engineering"},
        }
        call = SimpleNamespace(
            type="function_call", name="get_program_level_courses",
            arguments='{"level": 3}', call_id="level-3",
        )
        client = FakeClient([
            SimpleNamespace(id="r1", output=[call], output_text=""),
            SimpleNamespace(id="r2", output=[], output_text="Level 3 includes CIVL 3012."),
        ])

        with patch("ai_advisor.execute_tool", return_value={"level": 3, "courses": []}) as tool:
            result = answer_student_question(
                "what are all the level 3 courses for the civil engineering program?",
                [], client=client,
            )

        self.assertEqual(result["intent"], "PROGRAM_INFO")
        self.assertEqual(result["tools_used"], ["get_program_level_courses"])
        self.assertEqual(tool.call_args.args[2]["program_id"], "8660BENG")
        self.assertNotIn("which program", result["answer"].lower())

    def test_course_codes_are_normalized(self):
        self.assertEqual(normalize_course_id("civl 2020"), "CIVL2020")
        self.assertEqual(normalize_course_id("CIVL-2020"), "CIVL2020")

    @patch("ai_advisor.run_advisor_engine")
    @patch("ai_advisor.check_course_eligibility")
    def test_course_eligibility_uses_program_level_and_hides_history(
        self, prerequisite_check, advisor_engine
    ):
        prerequisite_check.return_value = {
            "course_id": "CIVL3012",
            "course_name": "Sustainability in Engineering",
            "eligible": True,
            "completed_courses": [{"course_id": "CIVL1012", "grade": 70}],
            "missing_requirements": [],
        }
        advisor_engine.return_value = {
            "current_level": 1,
            "evaluation_level": 2,
            "eligible_courses": [],
            "blocked_courses": [
                {
                    "course_id": "CIVL3012",
                    "course_name": "Sustainability in Engineering",
                    "missing_requirements": [
                        {"type": "PROGRAM_LEVEL", "required_level": 3}
                    ],
                }
            ],
        }
        context = {
            "program_id": "8660BENG",
            "completed_courses": [{"course_id": "CIVL1012", "grade": 70}],
        }

        result = execute_tool(
            "check_course_eligibility", {"course_id": "CIVL 3012"}, context
        )

        self.assertFalse(result["eligible"])
        self.assertEqual(result["missing_requirements"][0]["type"], "PROGRAM_LEVEL")
        self.assertNotIn("completed_courses", result)
        self.assertNotIn("grade", json.dumps(result))

    @patch("ai_advisor.check_course_eligibility")
    def test_course_eligibility_requires_program_for_full_verdict(
        self, prerequisite_check
    ):
        prerequisite_check.return_value = {
            "course_name": "Example Course",
            "eligible": True,
            "missing_requirements": [],
        }

        result = execute_tool(
            "check_course_eligibility",
            {"course_id": "TEST 1000"},
            {"program_id": None, "completed_courses": []},
        )

        self.assertIsNone(result["eligible"])
        self.assertTrue(result["prerequisites_satisfied"])
        self.assertIn("program", result["message"].lower())

    def test_plain_answer_does_not_send_a_catalog(self):
        client = FakeClient(
            [SimpleNamespace(id="r1", output=[], output_text="Please provide a program.")]
        )

        result = answer_student_question(
            "What can I take next?", [], client=client
        )

        request_text = client.responses.requests[0]["input"]
        self.assertNotIn("Course catalog", request_text)
        self.assertNotIn("completed_courses", request_text)
        self.assertEqual(result["tools_used"], [])

    def test_recent_conversation_is_available_for_follow_up_questions(self):
        client = FakeClient(
            [SimpleNamespace(id="r1", output=[], output_text="It is a core course.")]
        )

        answer_student_question(
            "Is it required?",
            [],
            conversation=[
                {"role": "user", "content": "Tell me about CIVL 2020."},
                {"role": "assistant", "content": "CIVL 2020 is Mechanics of Materials 1."},
            ],
            client=client,
        )

        request_text = client.responses.requests[0]["input"]
        self.assertIn("Recent conversation:", request_text)
        self.assertIn("CIVL 2020", request_text)

    def test_model_can_ask_for_missing_program_without_using_a_tool(self):
        client = FakeClient(
            [
                SimpleNamespace(
                    id="r1",
                    output=[],
                    output_text="Which program are you enrolled in?",
                )
            ]
        )

        result = answer_student_question(
            "What should I take next?", [], client=client
        )

        self.assertEqual(result["tools_used"], [])
        self.assertIn("program", result["answer"].lower())

    @patch("ai_advisor.execute_tool")
    def test_model_can_call_engine_tool_and_receive_structured_output(self, tool):
        tool.return_value = {
            "program_id": "8660BENG",
            "next_courses": [{"course_id": "CIVL2020"}],
        }
        call = SimpleNamespace(
            type="function_call",
            name="get_student_program_advice",
            arguments="{}",
            call_id="call-1",
        )
        client = FakeClient(
            [
                SimpleNamespace(id="r1", output=[call], output_text=""),
                SimpleNamespace(id="r2", output=[], output_text="You can take CIVL 2020."),
            ]
        )

        result = answer_student_question(
            "I finished level 1. What is next?",
            [{"course_id": "CIVL1012", "grade": 75}],
            program_id="8660beng",
            client=client,
        )

        self.assertEqual(result["tools_used"], ["get_student_program_advice"])
        self.assertEqual(result["answer"], "You can take CIVL 2020.")
        tool.assert_called_once()
        context = tool.call_args.args[2]
        self.assertEqual(context["program_id"], "8660BENG")

        follow_up = client.responses.requests[1]
        self.assertEqual(follow_up["previous_response_id"], "r1")
        supplied_result = json.loads(follow_up["input"][0]["output"])
        self.assertEqual(supplied_result["next_courses"][0]["course_id"], "CIVL2020")


if __name__ == "__main__":
    unittest.main(verbosity=2)
