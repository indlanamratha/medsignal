-- Where do reports come from? US vs. other countries.
select
    drug_group,
    count(*) as reports,
    round(100.0 * count(*) filter (where country = 'US') / count(*), 1) as pct_us,
    round(100.0 * count(*) filter (where country is not null and country <> 'US') / count(*), 1) as pct_non_us,
    round(100.0 * count(*) filter (where country is null) / count(*), 1) as pct_unknown
from fct_report_drug
group by drug_group
order by pct_non_us desc;
