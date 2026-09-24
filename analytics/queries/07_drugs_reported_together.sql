-- Which study drugs are most often suspects in the same report?
select a.drug_group as drug_a, b.drug_group as drug_b, count(*) as shared_reports
from fct_report_drug a
join fct_report_drug b
  on a.report_id = b.report_id
 and a.drug_group < b.drug_group
group by drug_a, drug_b
order by shared_reports desc
limit 15;
