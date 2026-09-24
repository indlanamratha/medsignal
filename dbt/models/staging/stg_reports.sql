-- One row per unique report, with readable labels.
select
    safetyreportid        as report_id,
    safetyreportversion   as report_version,
    receivedate           as received_date,
    receiptdate           as latest_receipt_date,
    is_serious,
    died,
    hospitalized,
    life_threatening,
    disabled,
    congenital_anomaly,
    other_serious,
    reporter_type,
    case reporter_type
        when 1 then 'Physician'
        when 2 then 'Pharmacist'
        when 3 then 'Other health professional'
        when 4 then 'Lawyer'
        when 5 then 'Consumer'
        else 'Unknown'
    end                   as reporter_type_label,
    country,
    age_years,
    case
        when age_years is null then 'Unknown'
        when age_years < 18 then '0-17'
        when age_years < 45 then '18-44'
        when age_years < 65 then '45-64'
        else '65+'
    end                   as age_group,
    case sex when 1 then 'M' when 2 then 'F' else 'U' end as sex,
    weight_kg
from read_parquet('{{ var("silver_path") }}/reports.parquet')
