"""MedSignal dashboard: FDA adverse event reports for obesity and diabetes drugs.

Run from the project folder:  uv run streamlit run dashboard/app.py
"""
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "medsignal.duckdb"

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
REPORTER_ORDER = ["Consumer", "Physician", "Other health professional", "Lawyer", "Unknown"]
REPORTER_COLORS = dict(zip(REPORTER_ORDER, [BLUE, ORANGE, AQUA, YELLOW, GRAY]))
AGE_ORDER = ["0-17", "18-44", "45-64", "65+", "Unknown"]

# Terms that describe how a product was used rather than a medical reaction.
NON_REACTION_FILTER = """
    and reaction_pt not like '%DOSE%'
    and reaction_pt not like '%ADMINIST%'
    and reaction_pt not like '%TECHNIQUE%'
    and reaction_pt not like 'PRODUCT %'
    and reaction_pt not in ('OFF LABEL USE', 'NO ADVERSE EVENT')
"""

DISCLAIMER = ("FAERS reports show that a reaction was reported after a drug was taken. "
              "They do not prove that the drug caused it.")


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(sql, list(params)).df()


def tidy(fig, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=10, r=30, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title=None))
    return fig


st.set_page_config(page_title="MedSignal", page_icon=":pill:", layout="wide")
st.title("MedSignal: Drug Safety Intelligence")
st.caption("FDA adverse event reports (FAERS) for 13 obesity and diabetes drug groups, "
           "received 2020–2025. Counts include suspect drugs only.")

# ---------------- Filters ----------------
classes = query("select distinct drug_class from dim_drug order by drug_class")["drug_class"].tolist()
with st.sidebar:
    st.header("Filters")
    sel_classes = st.multiselect("Drug class", classes, default=classes)
    year_from, year_to = st.slider("Year received", 2020, 2025, (2020, 2025))

if not sel_classes:
    st.warning("Select at least one drug class.")
    st.stop()

WHERE = "where list_contains(?, drug_class) and year(received_date) between ? and ?"
P = (sel_classes, year_from, year_to)

by_drug = query(f"""
    select drug_group, drug_class, count(*) as reports,
           round(100 * avg(is_serious::int), 1) as pct_serious,
           round(100 * avg((reporter_type_label = 'Consumer')::int), 1) as pct_consumer
    from fct_report_drug {WHERE}
    group by drug_group, drug_class
    order by reports
""", P)

if by_drug.empty:
    st.info("No reports match these filters.")
    st.stop()

drug_list = by_drug.sort_values("reports", ascending=False)["drug_group"].tolist()
tab_overview, tab_drug, tab_patterns = st.tabs(["Overview", "Drug explorer", "Reporting patterns"])

# ---------------- Overview ----------------
with tab_overview:
    kpi = query(f"""
        select count(*) as reports,
               round(100 * avg(is_serious::int), 1) as pct_serious,
               max(received_date) as latest
        from (select distinct report_id, is_serious, received_date from fct_report_drug {WHERE})
    """, P).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique reports", f"{int(kpi.reports):,}")
    c2.metric("Serious reports", f"{kpi.pct_serious}%")
    c3.metric("Drug groups", len(by_drug))
    c4.metric("Latest report received", pd.Timestamp(kpi.latest).strftime("%b %Y"))

    monthly = query(f"""
        select date_trunc('month', received_date) as month, count(distinct report_id) as reports
        from fct_report_drug {WHERE}
        group by month order by month
    """, P)
    left, right = st.columns(2)
    fig = px.line(monthly, x="month", y="reports", title="Reports received per month",
                  labels={"month": "", "reports": "Reports"}, color_discrete_sequence=[BLUE])
    fig.update_traces(line_width=2)
    left.plotly_chart(tidy(fig), width="stretch")

    fig = px.bar(by_drug, x="reports", y="drug_group", orientation="h",
                 title="Reports by drug", hover_data=["drug_class", "pct_serious"],
                 labels={"reports": "Reports", "drug_group": "", "pct_serious": "% serious",
                         "drug_class": "Class"},
                 color_discrete_sequence=[BLUE])
    right.plotly_chart(tidy(fig), width="stretch")

# ---------------- Drug explorer ----------------
with tab_drug:
    drug = st.selectbox("Drug", drug_list)
    hide_terms = st.checkbox("Hide medication-error and product-use terms", value=True,
                             help="Removes terms like 'Incorrect dose administered' that describe "
                                  "how a product was used rather than a medical reaction.")
    DW = "where drug_group = ? and year(received_date) between ? and ?"
    DP = (drug, year_from, year_to)

    row = by_drug.set_index("drug_group").loc[drug]
    c1, c2, c3 = st.columns(3)
    c1.metric("Reports", f"{int(row.reports):,}")
    c2.metric("Serious reports", f"{row.pct_serious}%")
    c3.metric("Filed by consumers", f"{row.pct_consumer}%")

    reactions = query(f"""
        select reaction_pt, count(distinct report_id) as reports
        from fct_drug_reaction {DW} {NON_REACTION_FILTER if hide_terms else ''}
        group by reaction_pt
        order by reports desc
        limit 15
    """, DP)
    reactions["pct_of_reports"] = (100 * reactions["reports"] / row.reports).round(1)

    left, right = st.columns([3, 2])
    fig = px.bar(reactions.sort_values("reports"), x="reports", y="reaction_pt", orientation="h",
                 title=f"Top 15 reactions: {drug}", hover_data=["pct_of_reports"],
                 labels={"reports": "Reports", "reaction_pt": "", "pct_of_reports": "% of reports"},
                 color_discrete_sequence=[BLUE])
    left.plotly_chart(tidy(fig, 480), width="stretch")

    ages = query(f"""
        select age_group, count(*) as reports, round(100 * avg(is_serious::int), 1) as pct_serious
        from fct_report_drug {DW}
        group by age_group
    """, DP)
    fig = px.bar(ages, x="age_group", y="reports", title="Reports by age group",
                 category_orders={"age_group": AGE_ORDER}, hover_data=["pct_serious"],
                 labels={"age_group": "", "reports": "Reports", "pct_serious": "% serious"},
                 color_discrete_sequence=[BLUE])
    right.plotly_chart(tidy(fig, 480), width="stretch")

# ---------------- Reporting patterns ----------------
with tab_patterns:
    mix = query(f"""
        select drug_group,
               case when reporter_type_label in ('Pharmacist', 'Other health professional')
                    then 'Other health professional' else reporter_type_label end as reporter,
               count(*) as reports
        from fct_report_drug {WHERE}
        group by all
    """, P)
    mix["pct"] = (100 * mix["reports"] / mix.groupby("drug_group")["reports"].transform("sum")).round(1)
    order = by_drug.sort_values("pct_consumer")["drug_group"].tolist()

    left, right = st.columns(2)
    fig = px.bar(mix, x="pct", y="drug_group", color="reporter", orientation="h",
                 title="Who files the reports (% of each drug's reports)",
                 category_orders={"reporter": REPORTER_ORDER, "drug_group": order},
                 color_discrete_map=REPORTER_COLORS,
                 labels={"pct": "% of reports", "drug_group": "", "reporter": "Reporter"})
    left.plotly_chart(tidy(fig, 460), width="stretch")

    fig = px.scatter(by_drug, x="pct_consumer", y="pct_serious", text="drug_group",
                     title="More consumer reports, fewer serious reports",
                     labels={"pct_consumer": "% filed by consumers", "pct_serious": "% serious"},
                     color_discrete_sequence=[BLUE])
    fig.update_traces(marker_size=10, textposition="top center", cliponaxis=False)
    right.plotly_chart(tidy(fig, 460), width="stretch")

    with st.expander("Show reporter mix as a table"):
        table = mix.pivot_table(index="drug_group", columns="reporter", values="pct", fill_value=0)
        st.dataframe(table.reindex(order[::-1]), width="stretch")

    st.subheader("Spike investigator")
    default = drug_list.index("semaglutide") if "semaglutide" in drug_list else 0
    spike_drug = st.selectbox("Drug", drug_list, index=default, key="spike_drug")
    spikes = query("""
        select date_trunc('month', received_date) as month,
               count(*) as "All reports",
               count(*) filter (where is_serious) as "Serious reports"
        from fct_report_drug
        where drug_group = ? and year(received_date) between ? and ?
        group by month order by month
    """, (spike_drug, year_from, year_to))
    long = spikes.melt(id_vars="month", var_name="series", value_name="reports")
    fig = px.line(long, x="month", y="reports", color="series",
                  title=f"Monthly reports vs. serious reports: {spike_drug}",
                  color_discrete_map={"All reports": BLUE, "Serious reports": ORANGE},
                  labels={"month": "", "reports": "Reports"})
    fig.update_traces(line_width=2)
    st.plotly_chart(tidy(fig), width="stretch")
    st.caption("A jump in all reports without a matching jump in serious reports usually points to "
               "a batch of mild reports filed at once, not a new safety problem.")

st.divider()
st.caption(f"{DISCLAIMER} Source: openFDA drug adverse event API (FAERS).")
