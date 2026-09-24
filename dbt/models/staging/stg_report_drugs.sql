-- One row per drug listed in a report.
select
    safetyreportid as report_id,
    medicinal_product,
    generic_name,
    brand_name,
    case drug_role
        when 1 then 'Suspect'
        when 2 then 'Concomitant'
        when 3 then 'Interacting'
        else 'Unknown'
    end as drug_role,
    indication
from read_parquet('{{ var("silver_path") }}/report_drugs.parquet')
