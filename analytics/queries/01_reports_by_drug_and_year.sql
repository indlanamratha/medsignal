-- How many reports did each drug receive per year (as a suspect drug)?
pivot (select drug_group, year(received_date) as yr from fct_report_drug)
on yr using count(*)
order by drug_group;
