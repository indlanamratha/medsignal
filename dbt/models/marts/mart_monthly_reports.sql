-- Report counts per drug group per month.
select
    drug_group,
    drug_class,
    date_trunc('month', received_date) as report_month,
    count(*) as n_reports,
    count(*) filter (where is_serious) as n_serious,
    round(100.0 * count(*) filter (where is_serious) / count(*), 1) as pct_serious
from {{ ref('fct_report_drug') }}
group by drug_group, drug_class, report_month
