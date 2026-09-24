-- Who reports each drug? Tests the reporting-bias idea.
select
    drug_group,
    count(*) as reports,
    round(100.0 * count(*) filter (where reporter_type_label = 'Consumer') / count(*), 1)  as pct_consumer,
    round(100.0 * count(*) filter (where reporter_type_label = 'Physician') / count(*), 1) as pct_physician,
    round(100.0 * count(*) filter (where reporter_type_label in ('Pharmacist', 'Other health professional')) / count(*), 1) as pct_other_hcp,
    round(100.0 * count(*) filter (where reporter_type_label = 'Lawyer') / count(*), 1)    as pct_lawyer,
    round(100.0 * avg(is_serious::int), 1) as pct_serious
from fct_report_drug
group by drug_group
order by pct_consumer desc;
