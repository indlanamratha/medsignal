-- One row per reaction listed in a report.
select distinct
    safetyreportid as report_id,
    reaction_pt,
    reaction_outcome
from read_parquet('{{ var("silver_path") }}/report_reactions.parquet')
where reaction_pt is not null
