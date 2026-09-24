-- Reactions reported much more often with GLP-1s than with other diabetes drugs.
-- (A simple preview of the signal statistics coming in Week 3.)
with glp1 as (
    select reaction_pt, count(distinct report_id) as n
    from fct_drug_reaction
    where drug_class in ('GLP-1', 'GLP-1/GIP')
    group by reaction_pt
),
other as (
    select reaction_pt, count(distinct report_id) as n
    from fct_drug_reaction
    where drug_class in ('Biguanide', 'SGLT2', 'DPP-4', 'Insulin')
    group by reaction_pt
),
glp1_total as (
    select count(distinct report_id) as t from fct_report_drug
    where drug_class in ('GLP-1', 'GLP-1/GIP')
),
other_total as (
    select count(distinct report_id) as t from fct_report_drug
    where drug_class in ('Biguanide', 'SGLT2', 'DPP-4', 'Insulin')
),
compared as (
    select g.reaction_pt, g.n as glp1_reports,
           round(100.0 * g.n / gt.t, 2) as pct_glp1,
           round(100.0 * coalesce(o.n, 0) / ot.t, 2) as pct_other
    from glp1 g
    left join other o using (reaction_pt)
    cross join glp1_total gt
    cross join other_total ot
    where g.n >= 200
)
select *, round(pct_glp1 / nullif(pct_other, 0), 1) as times_more_common
from compared
order by times_more_common desc nulls first
limit 20;
