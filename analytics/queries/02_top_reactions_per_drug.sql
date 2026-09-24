-- Top 10 reported reactions for each drug, with the share of that drug's reports.
with totals as (
    select drug_group, count(*) as drug_reports
    from fct_report_drug
    group by drug_group
),
counts as (
    select drug_group, reaction_pt, count(distinct report_id) as n_reports
    from fct_drug_reaction
    group by drug_group, reaction_pt
),
ranked as (
    select c.*, t.drug_reports,
           row_number() over (partition by c.drug_group order by c.n_reports desc) as rnk
    from counts c
    join totals t using (drug_group)
)
select drug_group, rnk, reaction_pt, n_reports,
       round(100.0 * n_reports / drug_reports, 1) as pct_of_drug_reports
from ranked
where rnk <= 10
order by drug_group, rnk;
