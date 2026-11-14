import os
import json
import unittest
import pandas as pd
import numpy as np
from Engine import engine
from dotenv import load_dotenv
from google import genai
from google.genai import types
from unittest.mock import Mock, patch

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    return genai.Client(api_key=API_KEY)

def parse_query_with_gemini(client, user_prompt):
    print(f"\n[AI] Sending prompt to Gemini: '{user_prompt}'")
    contents = (
        "Extract search constraints from this student college preference query. "
        "Return ONLY a valid JSON object with these keys: max_budget "
        "(integer or null), states (array of uppercase 2-letter state codes, "
        "or null when there is no location preference), major_keyword "
        "(string or null), and school_size (small, medium, large, or null). "
        "For a named US region, return all states in that region in the states "
        "array; for one requested state, return a one-item array. Do not return "
        "a region name instead of state codes. Use Census region definitions "
        "for Northeast, Midwest, South, and West. For school_size, use small "
        "for a small school, large for a big/large school, and medium only "
        "when the user asks for medium-sized schools. Query: "
        f"'{user_prompt}'"
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return response.text

def run_matcher(constraints, user_weights=[0.8, 0.6, 0.4]):
    selected_major = constraints.get("major_keyword") if constraints else None
    school_size = constraints.get("school_size") if constraints else None
    ideal_size_by_preference = {
        "small": 0.0,
        "medium": 0.5,
        "large": 1.0,
        "big": 1.0,
    }
    if school_size is None:
        ideal_size = 0.5
    else:
        preference = school_size.lower()
        if preference not in ideal_size_by_preference:
            raise ValueError(
                "school_size must be small, medium, large, big, or null"
            )
        ideal_size = ideal_size_by_preference[preference]

    matched_colleges = engine.match_colleges(
        user_weights,
        constraints,
        selected_major=selected_major,
        ideal_size=ideal_size,
    )
    return matched_colleges


class EnginePreferenceTests(unittest.TestCase):
    def test_parser_requests_state_lists_for_regions_and_school_size(self):
        response = Mock(text='{"states":["NC","SC"],"school_size":"small"}')
        client = Mock()
        client.models.generate_content.return_value = response

        result = parse_query_with_gemini(client, "a small school in the Southeast")

        self.assertEqual(json.loads(result)["states"], ["NC", "SC"])
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("all states in that region", prompt)
        self.assertIn("school_size", prompt)

    def test_matcher_passes_state_list_and_preferred_size_to_engine(self):
        with patch.object(engine, "match_colleges", return_value="matches") as match:
            for preference, expected_size in (("small", 0.0), ("big", 1.0)):
                constraints = {
                    "states": ["NC", "SC"],
                    "school_size": preference,
                    "major_keyword": None,
                }
                with self.subTest(preference=preference):
                    self.assertEqual(run_matcher(constraints), "matches")
                    self.assertEqual(match.call_args.args[1], constraints)
                    self.assertEqual(
                        match.call_args.kwargs["ideal_size"], expected_size
                    )

    def test_engine_filters_multiple_states_and_uses_ideal_size(self):
        institutions = pd.DataFrame(
            {
                "unit_id": [1, 2, 3],
                "school_name": ["A", "B", "C"],
                "state": ["NC", "SC", "NY"],
                "net_price": [10000, 20000, 30000],
                "graduation_rate": [0.5, 0.7, 0.9],
                "undergrad_size": [1000, 5000, 10000],
            }
        )
        with (
            patch.object(engine.pd, "read_csv", return_value=institutions),
            patch.object(
                engine,
                "cosine_similarity",
                side_effect=lambda matrix, vector: np.ones((len(matrix), 1)),
            ) as similarity,
        ):
            matches = engine.match_colleges(
                [1.0, 1.0, 1.0],
                {"states": ["NC", "SC"]},
                ideal_size=0.25,
            )

        self.assertEqual(set(matches["state"]), {"NC", "SC"})
        np.testing.assert_array_equal(
            similarity.call_args.args[1], [[0.0, 1.0, 0.25]]
        )

if __name__ == "__main__":
    print("========================================")
    print("   COLLEGE MATCHER BACKEND CLI TEST     ")
    print("========================================")
    
    # Initialize Client
    ai_client = get_gemini_client()
    
    # Sample test query
    test_prompt = "I want a big school in the Midwest under $40,000 with a computer science program."
    
    # Step A: Test Gemini Parsing
    json_output = parse_query_with_gemini(ai_client, test_prompt)
    print(f"\n[AI Parsed JSON Output]:\n{json_output}")
    
    try:
        parsed_constraints = json.loads(json_output)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON from Gemini response, using empty constraints.")
        parsed_constraints = {}
        
    # Step B: Test Vector Matching Engine
    # Weights: [Affordability: 1.0, Graduation Rate: 0.8, Size: 0.2]
    top_matches = run_matcher(parsed_constraints, user_weights=[1.0, 0.8, 0.2])
    
    # Step C: Display Formatted Results
    if top_matches is not None and not top_matches.empty:
        print("\n========================================")
        print("          TOP 10 MATCHING SCHOOLS        ")
        print("========================================")
        for _, row in top_matches.iterrows():
            print(f"🎓 {row['school_name']} ({row['state']})")
            print(f"   • Match Score    : {row['match_score']:.1f}%")
            roi = row["roi"]
            roi_display = (
                "N/A"
                if pd.isna(roi)
                else "∞ (zero debt)"
                if roi == float("inf")
                else f"{roi:.2f}x"
            )
            roi_score = row["roi_score"]
            roi_score_display = (
                f"{roi_score:.1f}" if pd.notna(roi_score) else "N/A"
            )
            print(f"   • ROI (earnings/debt): {roi_display}")
            print(f"   • ROI Score      : {roi_score_display}")
            print(f"   • Net Price      : ${row['net_price']:,.0f}")
            print(f"   • Graduation Rate: {row['graduation_rate']*100:.1f}%")
            print("-" * 40)
    else:
        print("\n No results returned.")