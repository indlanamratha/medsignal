-- One row per report x suspect drug group.
with suspect_drugs as (
    select distinct d.report_id, g.drug_group, g.drug_class
    from {{ ref('stg_report_drugs') }} as d
    join {{ ref('drug_groups') }} as g
      on d.generic_name like g.match_pattern
    where d.drug_role = 'Suspect'
)
select
    s.report_id,
    s.drug_group,
    s.drug_class,
    r.received_date,
    r.is_serious,
    r.died,
    r.hospitalized,
    r.age_group,
    r.sex,
    r.reporter_type_label,
    r.country
from suspect_drugs as s
join {{ ref('stg_reports') }} as r
  on r.report_id = s.report_id
