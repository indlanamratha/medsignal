-- Root cause of the report spikes: who filed the reports in the spike months?
select
    drug_group,
    strftime(date_trunc('month', received_date), '%Y-%m') as month,
    reporter_type_label,
    count(*) as reports,
    round(100.0 * avg(is_serious::int), 1) as pct_serious
from fct_report_drug
where (drug_group = 'semaglutide'
       and received_date between date '2025-06-01' and date '2025-08-31')
   or (drug_group = 'naltrexone/bupropion'
       and received_date between date '2025-10-01' and date '2025-12-31')
group by all
order by drug_group, month, reports desc;
