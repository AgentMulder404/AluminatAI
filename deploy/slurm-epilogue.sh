#!/bin/bash
# Slurm Epilog — AluminatAI job completion signal
#
# Signals job completion to the AluminatAI backend, which triggers the
# energy manifest generation DB trigger.
#
# slurm.conf example:
#   Epilog=/etc/slurm/epilog.d/aluminatai-epilogue.sh

set -euo pipefail

ALUMINATAI_API_KEY="${ALUMINATAI_API_KEY:-}"
ALUMINATAI_API_ENDPOINT="${ALUMINATAI_API_ENDPOINT:-https://aluminatiai.com/api/metrics/ingest}"
ENV_FILE="/run/aluminatai/job_${SLURM_JOB_ID}.env"

# Load job UUID from env file written by the prolog
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
    rm -f "$ENV_FILE"
fi

JOB_UUID="${ALUMINATAI_JOB_UUID:-}"

if [[ -z "$ALUMINATAI_API_KEY" || -z "$JOB_UUID" ]]; then
    exit 0
fi

# Signal job completion — triggers energy manifest generation in DB
curl -s -f -X POST \
    "${ALUMINATAI_API_ENDPOINT%/api/metrics/ingest}/api/metrics/jobs/complete" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${ALUMINATAI_API_KEY}" \
    -d "{
        \"job_id\": \"${JOB_UUID}\",
        \"end_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"exit_code\": ${SLURM_JOB_EXIT_CODE:-0}
    }" >/dev/null 2>&1 || true

exit 0
