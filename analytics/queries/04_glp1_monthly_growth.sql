-- Month-over-month growth in GLP-1 / GIP reports.
with monthly as (
    select report_month, sum(n_reports) as n_reports
    from mart_monthly_reports
    where drug_class in ('GLP-1', 'GLP-1/GIP')
    group by report_month
),
with_prev as (
    select report_month, n_reports,
           lag(n_reports) over (order by report_month) as prev_month
    from monthly
)
select strftime(report_month, '%Y-%m') as month, n_reports, prev_month,
       round(100.0 * (n_reports - prev_month) / prev_month, 1) as pct_change
from with_prev
order by report_month;
