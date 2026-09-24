-- Percentage of serious reports for each drug, by patient age group.
select
    drug_group,
    count(*) as reports,
    round(100.0 * avg(is_serious::int) filter (where age_group = '18-44'), 1)  as "18-44",
    round(100.0 * avg(is_serious::int) filter (where age_group = '45-64'), 1)  as "45-64",
    round(100.0 * avg(is_serious::int) filter (where age_group = '65+'), 1)    as "65+",
    round(100.0 * avg(is_serious::int) filter (where age_group = 'Unknown'), 1) as "age unknown"
from fct_report_drug
group by drug_group
order by reports desc;
