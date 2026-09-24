-- One row per drug group in the study.
select drug_group, drug_class
from {{ ref('drug_groups') }}
