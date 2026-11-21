# College Picker

College Picker is an explainable-AI project that matches students with U.S.
colleges using College Scorecard institution and field-of-study data. A Gemini
prompt turns a student's natural-language preferences into structured search
constraints. The Python engine filters eligible institutions, scores matches,
and returns up to 10 colleges.

**Deployed app:** https://explainableai-collegepicker.streamlit.app/

## Goal

The goal of this project was to explore the power of white-box and simple ML approaches on complex problems. The engine runs using simple  constraint filtering and vector matching to find the best schools for a given user prompt.

**Conclusions:** White-box algorithms can be extremely powerful, even for complex problems such as finding the right college. By allowing users to see as much information into the algorithm, they are able to make a much more informed decision and understand exactly why a school was recommended to them.

**Note:** This project was not created to denounce black-box algorithms entirely. These models continue to be extremely powerful and are greatly needed in complex environments. This project simply looks at the impact of interpretability by using an entirely white-box approach over a potentially simpler black-box.

## Current features

- Parse college preferences with Gemini, including states or Census regions,
  school size, major, and prestige emphasis.
- Filter by institution level (4-year, 2-year, or under 2-year), graduation
  rate, 10-year median earnings, debt at graduation, and 3-year repayment rate.
  Numeric filters support minimum and maximum bounds. For example:
  `min_graduation_rate: 0.7` means a graduation rate of at least 70%.
- Match on affordability, graduation rate, and undergraduate size using a
  weighted cosine-similarity score.
- Optionally filter out schools whose reported average SAT score is more than
  100 points above the student's SAT score.
- Optionally filter out schools whose reported ACT score is more than 5 points
  above the student's ACT score.
- Calculate a prestige score from available admission rate, SAT score,
  instructional expenditure, and Carnegie classification data, then combine it
  with the match score using the requested prestige weight.
- When a major is specified, restrict results to institutions offering a
  matching bachelor's program and calculate a program-level earnings-to-debt
  ROI score where data is available.

The engine returns the top 10 results with scores and source columns for
inspection. The top match gets an additional XAI audit for users to allow for more interpretability.
The repository includes a Streamlit web interface and a command-line Gemini example.

## Project layout

```text
Data/
  data_source.md                         Data sources and field mappings
  process_data.py                        Creates the cleaned CSV datasets
  institutions_clean.csv                 Clean institution-level records
  programs_clean.csv                     Clean bachelor's program records
app.py                                    Streamlit web interface
Engine/
  engine.py                              Filtering, scoring, and ranking
test_engine.py                           Unit tests and Gemini CLI example
requirements.txt                         Python dependencies
```

Raw and cleaned CSV files are excluded from Git. See [Data/data_source.md](Data/data_source.md)
for source details and the raw-to-clean field mappings.

## Setup

Use Python 3 and install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the latest institution and field-of-study CSVs from the [College
Scorecard data page](https://collegescorecard.ed.gov/data/) and place them in
`Data/` with these filenames:

- `Most-Recent-Cohorts-Institution.csv`
- `Most-Recent-Cohorts-Field-of-Study.csv`

Generate the cleaned datasets from the repository root:

```bash
python Data/process_data.py
```

To use the Gemini example, add an API key to a root-level `.env` file:

```dotenv
GEMINI_API_KEY=your_api_key
```

The `.env` file is ignored by Git. Run the example with:

```bash
python test_engine.py
```

The example sends a sample query to Gemini and displays the resulting college
matches. `Engine.engine.match_colleges(...)` can also be called directly with
parsed constraints.

## Testing

Run the unit tests without making a Gemini API call:

```bash
python -m unittest test_engine -v
```

## Constraint conventions

The Gemini parser returns JSON containing optional constraint fields. Use
`institution_levels` as an array of numeric codes (`1`, `2`, or `3`).
Graduation and repayment rate bounds use fractions from 0 to 1; dollar-value
bounds use numeric amounts. Unspecified bounds should be `null`. Suppressed or
nonnumeric earnings, debt, or repayment values do not match a constraint on
that field.

The `max_budget` field is present in the Gemini prompt schema, but a budget
hard-filter is not currently implemented in the matching engine.

## Limitations and Future Directions

1. Net Price is currently aggregated over all income bands. Future iterations will allow the user to optionally input their family income for a more accurate net price.
2. The Gemini API occasionally reports constraints inaccurately. This is already visible to users through the "View Parsed AI Constraints" view.
3. Matching is currently limited to major names in the College Scorecard. Future iterations will use a stronger LLM to also search similar majors outside of the explicit one stated by the user.

## References

1. https://collegescorecard.ed.gov/

---

Developer: Rahul Govil