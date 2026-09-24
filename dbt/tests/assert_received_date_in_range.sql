-- Every report must have been received in the study window (2020-2025).
-- dbt treats any returned row as a failure.
select report_id, received_date
from {{ ref('stg_reports') }}
where received_date < date '2020-01-01'
   or received_date > date '2025-12-31'
