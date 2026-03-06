#!/bin/bash
# Slurm Prolog — AluminatAI job attribution
#
# Drop this file in the Slurm prolog directory (PrologSlurmctld or Prolog in slurm.conf).
# It exports ALUMINATAI_JOB_UUID into the job environment so the agent can tag
# every metric row with the correct job UUID for chargeback attribution.
#
# slurm.conf example:
#   Prolog=/etc/slurm/prolog.d/aluminatai-prologue.sh

set -euo pipefail

ALUMINATAI_API_KEY="${ALUMINATAI_API_KEY:-}"
ALUMINATAI_API_ENDPOINT="${ALUMINATAI_API_ENDPOINT:-https://aluminatiai.com/api/metrics/ingest}"

# Only run if API key is configured
if [[ -z "$ALUMINATAI_API_KEY" ]]; then
    exit 0
fi

# Register the job with AluminatAI and get back a job UUID
JOB_UUID=$(curl -s -f -X POST \
    "${ALUMINATAI_API_ENDPOINT%/api/metrics/ingest}/api/metrics/jobs/register" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${ALUMINATAI_API_KEY}" \
    -d "{
        \"slurm_job_id\": \"${SLURM_JOB_ID}\",
        \"job_name\": \"${SLURM_JOB_NAME}\",
        \"user\": \"${SLURM_JOB_USER}\",
        \"account\": \"${SLURM_JOB_ACCOUNT}\",
        \"partition\": \"${SLURM_JOB_PARTITION}\",
        \"num_gpus\": ${SLURM_GPUS_ON_NODE:-0}
    }" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_uuid',''))" 2>/dev/null || echo "")

if [[ -n "$JOB_UUID" ]]; then
    # Export into the job environment (Slurm exports prolog env to the job)
    export ALUMINATAI_JOB_UUID="$JOB_UUID"

    # Also write to a job-specific env file that the agent can source
    echo "ALUMINATAI_JOB_UUID=$JOB_UUID" > "/run/aluminatai/job_${SLURM_JOB_ID}.env"
    echo "ALUMINATAI_TEAM_ID=${SLURM_JOB_ACCOUNT:-default}" >> "/run/aluminatai/job_${SLURM_JOB_ID}.env"
fi

exit 0
