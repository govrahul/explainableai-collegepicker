import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
import json

# Import your matching engine module
from Engine import engine

# 1. Page Configuration
st.set_page_config(
    page_title="AI College & Program Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Secure Gemini Client Initialization
@st.cache_resource
def get_gemini_client():
    """Initializes the Gemini client securely using Streamlit secrets or environment fallback."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in Streamlit secrets or environment variables.")
        st.stop()
    return genai.Client(api_key=api_key)

ai_client = get_gemini_client()

# 3. Prompt Parsing Function (Matches your test script logic)
def parse_query_with_gemini(client, user_prompt):
    contents = (
        "Extract search constraints from this student college preference query. "
        "Return ONLY a valid JSON object with these keys: max_budget "
        "(integer or null), states (array of uppercase 2-letter state codes, "
        "or null when there is no location preference), major_keyword "
        "(string or null), school_size (small, medium, large, or null), and "
        "prestige (number from 0.0 to 1.0, used as the ranking weight for "
        "prestige versus match score). Use 0.5 when there is no explicit "
        "prestige preference; use a higher value when the user asks for an "
        "elite, prestigious, selective, renowned, or top-tier school, with "
        "stronger emphasis reflected by a higher value. "
        "Also return min_median_earnings_10yr and max_median_earnings_10yr "
        "(numeric dollar amounts or null), institution_levels (an array "
        "containing any requested institution level codes or null), "
        "min_graduation_rate and max_graduation_rate (numbers from 0.0 to 1.0 "
        "or null), "
        "min_median_debt_at_graduation and max_median_debt_at_graduation "
        "(numeric dollar amounts or null), and min_repayment_within_3_years "
        "and max_repayment_within_3_years (numbers from 0.0 to 1.0 or null). "
        "Only set a minimum or maximum when the user explicitly requests that "
        "bound; otherwise use null. Convert graduation-rate and repayment "
        "percentages to fractions (for example, 70 percent becomes 0.7). "
        "Map 4-year institutions to "
        "level 1, 2-year institutions to level 2, and institutions under 2 "
        "years to level 3. If the user does not specify a level, use null. "
        "For a named US region, return all states in that region in the states "
        "array; for one requested state, return a one-item array. Do not return "
        "a region name instead of state codes. Use Census region definitions "
        "for Northeast, Midwest, South, and West. For school_size, use small "
        "for a small school, large for a big/large school, and medium only "
        "when the user asks for medium-sized schools. Query: "
        f"'{user_prompt}'"
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return response.text

# 4. Programmatic XAI Generator
def generate_xai_breakdown(row):
    """Dynamically builds XAI explanations from row metrics without hardcoded templates."""
    strengths = []
    weaknesses = []
    
    if row.get('net_price', 0) > 0 and row.get('net_price', 0) <= 30000:
        strengths.append(f"strong affordability (Net Price: ${row['net_price']:,.0f})")
    elif row.get('net_price', 0) > 50000:
        weaknesses.append(f"higher net price (${row['net_price']:,.0f})")
        
    if row.get('graduation_rate', 0) >= 0.75:
        strengths.append(f"high graduation success ({row['graduation_rate']*100:.1f}%)")
    elif row.get('graduation_rate', 0) < 0.50 and pd.notna(row.get('graduation_rate')):
        weaknesses.append(f"lower completion rate ({row['graduation_rate']*100:.1f}%)")
        
    if row.get('prestige_score', 0) >= 70:
        strengths.append(f"elite selectivity and institutional standing (Prestige: {row['prestige_score']:.1f})")

    strength_str = ", ".join(strengths) if strengths else "balanced baseline metrics"
    weakness_str = f" However, it traded off on {', '.join(weaknesses)}." if weaknesses else ""
    
    return f"This institution ranked high primarily due to its {strength_str}.{weakness_str}"

# 5. Plotly Delta Bar Chart for XAI Visuals
def create_delta_bar_chart(row):
    """Renders a clean horizontal breakdown of key normalized vector features."""
    categories = ['Affordability (Net Price)', 'Graduation Success', 'Undergrad Scale']
    values = [
        row.get('norm_price', 0.5) * 100, 
        row.get('norm_grad', 0.5) * 100, 
        row.get('norm_size', 0.5) * 100
    ]
    
    fig = px.bar(
        x=values,
        y=categories,
        orientation='h',
        text=[f"{v:.1f}%" for v in values],
        range_x=[0, 100]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=140,
        xaxis=dict(showgrid=True, title="Normalized Feature Score"),
        yaxis=dict(title="")
    )
    return fig

# 6. Streamlit User Interface Layout
st.title("AI College & Program Recommender")
st.markdown("Transparent, vector-based matching powered by real U.S. Department of Education College Scorecard data.")

# Sidebar Configuration
st.sidebar.header("Custom Vector Weights & Preferences")

# Clear explainer popover addressing the "sum to 1" question
with st.sidebar.popover("ℹ️ How do these weights work? (Do they need to sum to 1?)"):
    st.markdown("""
    **No, they do not need to sum to 1.0!**
    
    Your weights act as independent multipliers on each feature vector:
    * **Setting them all to 1.0** simply gives equal, full importance to affordability, graduation rate, and campus size.
    * **Independent scaling:** You can set all sliders to max (1.0) or zero out a feature you don't care about without worrying about percentages or math rules. Cosine similarity evaluates the directional alignment automatically.
    """)

weight_affordability = st.sidebar.slider("Affordability Weight", 0.0, 1.0, 1.0, step=0.1, 
                                         help="Higher values heavily penalize schools that have a higher net price, ranking lower-cost options closer to your ideal score.")
weight_grad = st.sidebar.slider("Graduation Rate Weight", 0.0, 1.0, 0.8, step=0.1,
                                help="Higher values reward schools with high completion and graduation rates in the vector similarity calculation.")
weight_size = st.sidebar.slider("College Size Weight", 0.0, 1.0, 0.2, step=0.1,
                                help="Controls how strictly the model matches your preferred campus size (small vs. large) against your target preference.")
user_weights = [weight_affordability, weight_grad, weight_size]

st.sidebar.divider()
st.sidebar.markdown("### Natural Language Search")

filter_by_sat = st.sidebar.checkbox(
    "Apply SAT score filter",
    help=(
        "When enabled, only schools with a reported average SAT score no more "
        "than 100 points above yours are included. Schools without SAT data "
        "are excluded."
    ),
)
user_sat_score = st.sidebar.number_input(
    "Your SAT score",
    min_value=400,
    max_value=1600,
    value=None,
    step=10,
    disabled=not filter_by_sat,
    placeholder="Enter your score",
)
filter_by_act = st.sidebar.checkbox(
    "Apply ACT score filter",
    help=(
        "When enabled, only schools with a reported ACT score no more than "
        "5 points above yours are included. Schools without ACT data are excluded."
    ),
)
user_act_score = st.sidebar.number_input(
    "Your ACT score",
    min_value=1,
    max_value=36,
    value=None,
    step=1,
    disabled=not filter_by_act,
    placeholder="Enter your score",
)

# Info Popover for Filter Capabilities
with st.sidebar.popover("ℹ️ What can you filter?"):
    st.markdown("### Intelligent Query Guide")
    st.markdown("""
    Gemini automatically parses your description into the following filter criteria. You can ask for:
    * **📍 Location:** Specific states (e.g., *"in NC or SC"*) or Census regions (*Northeast, Midwest, South, West*).
    * **💰 Budget:** Maximum annual net price (e.g., *"under $40k"*).
    * **🎓 Academics & Majors:** Specific fields of study or keywords (e.g., *"Computer Science"*).
    * **🏛️ Institution Level:** 4-year, 2-year, or less than 2-year colleges.
    * **📈 Success & Outcomes:** 
        * Minimum/maximum graduation rates (e.g., *"at least 70% grad rate"*).
        * 10-year median post-grad earnings thresholds.
        * Maximum student loan debt at graduation (e.g., *"at most $20,000 debt at graduation"*).
        * 3-year loan repayment rates (e.g., *"at least 20% loan repayment after 3 years"*).
    * **⭐ Prestige & Selectivity:** Requesting *elite, prestigious, selective, or top-tier* schools to weight institutional rank.
    * **🏫 Campus Size:** Small, medium, or large student populations.
    """)

default_query = st.sidebar.text_area(
    "Describe what you're looking for:",
    value="Top-tier computer science program under $45k in the Northeast",
    help="Gemini parses your query into strict constraints and prestige weighting automatically."
)

if st.sidebar.button("🔍 Run Intelligent Match", type="primary"):
    if filter_by_sat and user_sat_score is None:
        st.error("Enter your SAT score to use the SAT score filter.")
        st.stop()
    if filter_by_act and user_act_score is None:
        st.error("Enter your ACT score to use the ACT score filter.")
        st.stop()

    with st.spinner("🤖 Parsing query with Gemini and calculating vector distances..."):
        try:
            # Step 1: Parse query via Gemini JSON mode
            json_output = parse_query_with_gemini(ai_client, default_query)
            constraints = json.loads(json_output)
            if filter_by_sat:
                constraints["max_average_sat_score"] = user_sat_score + 100
            if filter_by_act:
                constraints["max_average_act_score"] = user_act_score + 5
            
            # Extract parameters for engine
            selected_major = constraints.get("major_keyword")
            school_size = constraints.get("school_size")
            prestige_weight = constraints.get("prestige", 0.5)
            
            ideal_size_map = {"small": 0.0, "medium": 0.5, "large": 1.0, "big": 1.0}
            ideal_size = ideal_size_map.get(school_size.lower(), 0.5) if school_size else 0.5

            # Step 2: Execute backend matching engine
            top_matches = engine.match_colleges(
                user_weights=user_weights,
                constraints=constraints,
                selected_major=selected_major,
                ideal_size=ideal_size,
                prestige_weight=prestige_weight
            )
            
            # Store in session state for persistence
            st.session_state['top_matches'] = top_matches
            st.session_state['constraints'] = constraints
            
        except Exception as e:
            st.error(f"❌ Error during matching execution: {e}")
            st.stop()

# 7. Render Results Dashboard
if 'top_matches' in st.session_state:
    matches = st.session_state['top_matches']
    constraints_used = st.session_state.get('constraints', {})
    
    with st.expander("🛠️ View Parsed AI Constraints"):
        st.json(constraints_used)
        
    if matches is None or matches.empty:
        st.warning("⚠️ No matching institutions found meeting your strict constraints. Try broadening your location or budget filters.")
    else:
        st.success(f"✨ Found {len(matches)} matching institutions ranked by your custom vector model!")
        st.divider()
        
        for _, row in matches.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.subheader(f"🎓 {row['school_name']} ({row['state']})")
                    st.caption(f"Institution ID: `{row['unit_id']}`")
                with cols[1]:
                    st.metric("Match Score", f"{row['ranking_score']:.1f}%")
                with cols[2]:
                    net_price = row.get('net_price', 0)
                    st.metric("Net Price", f"${net_price:,.0f}" if pd.notna(net_price) else "N/A")
                
                # Secondary metric pills row
                stat_cols = st.columns(4)
                grad_rate = row.get('graduation_rate', 0)
                stat_cols[0].markdown(f"**Graduation Rate:** {grad_rate*100:.1f}%" if pd.notna(grad_rate) else "**Grad Rate:** N/A")
                
                prestige = row.get('prestige_score', 0)
                stat_cols[1].markdown(f"**Prestige Score:** {prestige:.1f}")
                
                roi = row.get('roi')
                if pd.isna(roi):
                    roi_str = "N/A"
                elif roi == float("inf"):
                    roi_str = "∞ (zero debt)"
                else:
                    roi_str = f"{roi:.2f}x"
                stat_cols[2].markdown(f"**Program ROI:** {roi_str}")
                
                level_map = {1: "4-Year", 2: "2-Year", 3: "< 2-Year"}
                lvl = row.get('institution_level')
                stat_cols[3].markdown(f"**Level:** {level_map.get(lvl, 'Unknown')}")

                # Programmatic XAI Expander & Plotly Chart
                with st.expander("🔍 Inspect Match Breakdown & Trade-offs"):
                    # Programmatic narrative sentence
                    xai_narrative = generate_xai_breakdown(row)
                    st.write(xai_narrative)
                    
                    st.markdown("##### Feature Normalization Contributions")
                    fig = create_delta_bar_chart(row)
                    st.plotly_chart(fig, width='stretch', key=f"chart_{row['unit_id']}")
                    
            st.markdown("---")