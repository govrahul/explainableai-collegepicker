import os
import json
import pandas as pd
from Engine import engine
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

def get_gemini_client():
    return genai.Client(api_key=API_KEY)

def parse_query_with_gemini(client, user_prompt):
    print(f"\n[AI] Sending prompt to Gemini: '{user_prompt}'")
    contents = f"Extract search constraints from this student college preference query. Return ONLY a valid JSON object with keys: max_budget (integer or null), state (2-letter abbreviation uppercase or null), major_keyword (string or null). Query: '{user_prompt}'"

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
    matched_colleges = engine.match_colleges(
        user_weights, constraints, selected_major=selected_major
    )
    return matched_colleges

if __name__ == "__main__":
    print("========================================")
    print("   COLLEGE MATCHER BACKEND CLI TEST     ")
    print("========================================")
    
    # Initialize Client
    ai_client = get_gemini_client()
    
    # Sample test query
    test_prompt = "I want an affordable school in North Carolina under $35,000 with a computer science program."
    
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