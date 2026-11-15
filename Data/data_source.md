# Data Sources

The project uses data from the U.S. Department of Education's [College
Scorecard](https://collegescorecard.ed.gov/data/). The full data dictionary is
available in `CollegeScorecardDataDictionary.xlsx` and on the [College Scorecard
documentation page](https://collegescorecard.ed.gov/data/data-documentation/).

`process_data.py` reads the institution-level and field-of-study CSVs and writes
the cleaned datasets `institutions_clean.csv` and `programs_clean.csv`.

## Institution-level data

Source file: `Most-Recent-Cohorts-Institution.csv`

| Raw field | Clean field | Description |
| --- | --- | --- |
| `UNITID` | `unit_id` | Institution identifier |
| `INSTNM` | `school_name` | Institution name |
| `STABBR` | `state` | State abbreviation |
| `REGION` | `region_code` | Census region code |
| `LOCALE` | `locale_code` | Locale code |
| `NPT4_PUB`, `NPT4_PRIV` | `net_price` | Public net price when available; otherwise private net price |
| `C150_4` | `graduation_rate` | Reported graduation/completion rate |
| `UGDS` | `undergrad_size` | Undergraduate enrollment |
| `ADM_RATE` | `admission_rate` | Admission rate |
| `SAT_AVG` | `average_sat_score` | Average SAT score |
| `ACTCMMID` | `average_act_score` | ACT composite midpoint |
| `CCBASIC` | `carnegie_classification` | Carnegie classification |
| `INEXPFTE` | `instructional_expenditure_per_student` | Instructional expenditure per student |
| `MD_EARN_WNE_P10` | `median_earnings_10yr` | Median earnings 10 years after entry |
| `ICLEVEL` | `institution_level` | Institution level: 1 = 4-year, 2 = 2-year, 3 = less than 2-year |
| `GRAD_DEBT_MDN` | `median_debt_at_graduation` | Median debt at graduation |
| `COMPL_RPY_3YR_RT` | `repayment_within_3_years` | Share of borrowers who completed repayment within 3 years |

Graduation and repayment rates are represented as fractions from 0 to 1. The
engine accepts optional minimum and maximum rate constraints; percentage values
from a user's query are converted to fractions by the Gemini prompt.

The earnings, debt, and repayment source fields may contain the College
Scorecard suppression value `PS`. The matcher treats nonnumeric values as
missing for a metric when the user filters on that metric.

## Field-of-study data

Source file: `Most-Recent-Cohorts-Field-of-Study.csv`. The preprocessing script
keeps bachelor's degree records (`CREDLEV == 3`) and uses these fields:

| Raw field | Clean field | Description |
| --- | --- | --- |
| `UNITID` | `unit_id` | Institution identifier |
| `INSTNM` | `school_name` | Institution name |
| `CIPCODE` | `major_code` | Classification of Instructional Programs code |
| `CIPDESC` | `major_name` | Major/program name |
| `EARN_MDN_HI_1YR` | `median_earnings_1yr` | Median earnings one year after completion |
| `DEBT_ALL_STGP_EVAL_MDN` | `median_federal_debt` | Median federal debt for program completers |

`CREDLEV` is used to select bachelor's degree records and is then dropped from
the cleaned output. Suppressed `PS` values for the program earnings and debt
metrics are converted to missing values.
