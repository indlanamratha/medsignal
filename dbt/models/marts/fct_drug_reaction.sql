-- One row per report x suspect drug group x reaction.
select distinct
    f.report_id,
    f.drug_group,
    f.drug_class,
    rx.reaction_pt,
    f.received_date,
    f.is_serious
from {{ ref('fct_report_drug') }} as f
join {{ ref('stg_report_reactions') }} as rx
  on rx.report_id = f.report_id
