#!/usr/bin/env bash
# Release pipeline for the 3-tier AWS setup (web / api / db servers behind an ALB), run from your PC:
#   ./deploy/deploy.sh <path/to/private-key> [stage ...]
#
# Stages (default: all, in this order):
#   setup  - check local prerequisites, read server addresses from Terraform, prepare the 3 servers
#   build  - build the api, web and test images locally with one release tag
#   test   - run the backend test suite locally against a throwaway Postgres (no API credits)
#   deploy - db up -> ship api image + documents, sync knowledge base, switch api -> ship + switch web
#   smoke  - check the live site through the load balancer and that no tier is reachable directly
# If deploy or smoke fails after a tier was switched, that tier is rolled back automatically.
#
# Server addresses come from `infra/tf.sh output`; set INVENTORY=<file> to supply them yourself.
set -euo pipefail

ALL_STAGES=(setup build test deploy smoke)
usage() {
  echo "Usage: $0 <path/to/private-key> [${ALL_STAGES[*]}]" >&2
  exit 1
}
[[ $# -ge 1 ]] || usage

KEY="$1"
shift
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=("${ALL_STAGES[@]}")
for s in "${STAGES[@]}"; do
  [[ " ${ALL_STAGES[*]} " == *" $s "* ]] || { echo "Unknown stage: $s" >&2; usage; }
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=10)
[[ -n "${SSH_PORT:-}" ]] && SSH_OPTS+=(-o Port="$SSH_PORT")
REFUSAL="The information is not available in the current documents."
LAST_BUILD_FILE="$ROOT/deploy/.last_build"

bold=$'\e[1m'; green=$'\e[32m'; red=$'\e[31m'; yellow=$'\e[33m'; reset=$'\e[0m'
log() { echo "${bold}$*${reset}"; }
ok() { echo "  ${green}✓${reset} $*"; }
fail() { echo "  ${red}✗${reset} $*" >&2; return 1; }
json_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

on() { local host="$1"; shift; ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$host" "$@"; }
remote() { local host="$1"; shift; on "$host" "bash -s -- $*" < "$ROOT/deploy/remote.sh"; }
upload() { on "$1" "umask 077 && cat > aims-rag/$2"; }

load_inventory() {
  local inv="${INVENTORY:-$ROOT/deploy/inventory.env}"
  if [[ -z "${INVENTORY:-}" ]]; then
    local outputs
    outputs="$("$ROOT/infra/tf.sh" output -json)" || fail "could not read Terraform outputs; run ./infra/tf.sh apply first"
    TF_OUTPUTS="$outputs" python3 - > "$inv" <<'PY' || fail "Terraform outputs incomplete; run ./infra/tf.sh apply first"
import json, os
o = json.loads(os.environ["TF_OUTPUTS"])
servers = o["servers"]["value"]
print("APP_URL=" + o["app_url"]["value"])
for role in ("web", "api", "db"):
    print(role.upper() + "_HOST=" + servers[role]["public_ip"])
    print(role.upper() + "_PRIVATE=" + servers[role]["private_ip"])
PY
  fi
  # shellcheck disable=SC1090
  source "$inv"
  local v
  for v in APP_URL WEB_HOST API_HOST DB_HOST WEB_PRIVATE API_PRIVATE DB_PRIVATE; do
    [[ -n "${!v:-}" ]] || fail "$v missing from $inv"
  done
}

ship_image() {
  local host="$1" image="$2"
  if remote "$host" has-image "$image"; then
    ok "$image already on $host"
    return
  fi
  docker save "$image" | gzip -1 | on "$host" 'gunzip | sudo docker load -q' >/dev/null
  ok "$image shipped to $host"
}

release_tag() {
  if [[ -n "${IMAGE_TAG:-}" ]]; then
    echo "$IMAGE_TAG"
  elif [[ -f "$LAST_BUILD_FILE" ]]; then
    cat "$LAST_BUILD_FILE"
  else
    fail "no build found; run the build stage first"
  fi
}

stage_setup() {
  local cmd
  for cmd in ssh rsync docker gzip curl python3; do
    command -v "$cmd" >/dev/null || fail "missing local command: $cmd"
  done
  ok "local tools present"
  [[ -f "$KEY" ]] || fail "key not found: $KEY"
  [[ "$(stat -c %a "$KEY")" =~ ^[46]00$ ]] || fail "key permissions too open; run: chmod 400 $KEY"
  ok "ssh key ok"
  [[ -f "$ROOT/backend/.env" ]] || fail "backend/.env missing (copy backend/.env.example)"
  grep -qE '^OPENROUTER_API_KEY=.+' "$ROOT/backend/.env" || fail "OPENROUTER_API_KEY empty in backend/.env"
  grep -qE '^POSTGRES_PASSWORD=.+' "$ROOT/backend/.env" || fail "POSTGRES_PASSWORD empty in backend/.env"
  ok "backend/.env has required keys"
  local docs
  docs="$(find "$ROOT/backend/data/documents" -name '*.json' 2>/dev/null | wc -l)"
  (( docs > 0 )) || fail "no documents in backend/data/documents"
  ok "$docs documents ready"
  load_inventory
  ok "inventory: web=$WEB_HOST api=$API_HOST db=$DB_HOST"
  local role host
  for role in web api db; do
    host="$(eval echo "\$${role^^}_HOST")"
    remote "$host" setup >/dev/null || fail "$role server $host not reachable or not ready (ssh allowed from your current IP?)"
    ok "$role server ready ($host)"
  done
}

stage_build() {
  local tag
  tag="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
  [[ -z "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]] || tag="$tag-dirty"
  tag="${IMAGE_TAG:-$tag-$(date -u +%Y%m%d%H%M%S)}"
  (cd "$ROOT" && IMAGE_TAG="$tag" docker compose build api web)
  (cd "$ROOT" && IMAGE_TAG="$tag" docker compose -f docker-compose.test.yml build tests)
  echo "$tag" > "$LAST_BUILD_FILE"
  ok "built aims-rag-api:$tag, aims-rag-web:$tag, aims-rag-api-test:$tag"
}

stage_test() {
  local tag
  tag="$(release_tag)"
  (
    cd "$ROOT"
    trap 'IMAGE_TAG="$tag" docker compose -f docker-compose.test.yml down -v >/dev/null 2>&1' EXIT
    IMAGE_TAG="$tag" docker compose -f docker-compose.test.yml run --rm tests
  )
  ok "test suite passed"
}

DEPLOYED_API=0
DEPLOYED_WEB=0

stage_deploy() {
  local tag
  tag="$(release_tag)"
  load_inventory

  log "  db tier ($DB_HOST)"
  upload "$DB_HOST" compose.yml < "$ROOT/deploy/compose/db.yml"
  grep -E '^POSTGRES_PASSWORD=' "$ROOT/backend/.env" | upload "$DB_HOST" .env
  remote "$DB_HOST" db-up
  ok "postgres healthy"

  log "  api tier ($API_HOST)"
  ship_image "$API_HOST" "aims-rag-api:$tag"
  on "$API_HOST" "mkdir -p aims-rag/data/documents"
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$ROOT/backend/data/documents/" "$REMOTE_USER@$API_HOST:aims-rag/data/documents/"
  ok "documents synced"
  upload "$API_HOST" compose.yml < "$ROOT/deploy/compose/api.yml"
  { grep -vE '^(POSTGRES_HOST|DATABASE_URL)=' "$ROOT/backend/.env"; echo "POSTGRES_HOST=$DB_PRIVATE"; } | upload "$API_HOST" .env
  DEPLOYED_API=1
  remote "$API_HOST" release api "$tag"
  ok "api live on $tag"

  log "  web tier ($WEB_HOST)"
  ship_image "$WEB_HOST" "aims-rag-web:$tag"
  upload "$WEB_HOST" compose.yml < "$ROOT/deploy/compose/web.yml"
  printf 'API_URL=http://%s:8000\nRATE_LIMIT_PER_MINUTE=%s\n' "$API_PRIVATE" "${RATE_LIMIT_PER_MINUTE:-10}" | upload "$WEB_HOST" .env
  DEPLOYED_WEB=1
  remote "$WEB_HOST" release web "$tag"
  ok "web live on $tag"

  remote "$API_HOST" prune aims-rag-api
  remote "$WEB_HOST" prune aims-rag-web
}

stage_smoke() {
  load_inventory
  local html="" body code answer sources chunks
  # The ALB needs two passing health checks (~30s) after a new target comes up.
  for _ in $(seq 1 20); do
    html="$(curl -fsS -m 10 "$APP_URL/" 2>/dev/null)" && break
    sleep 5
  done
  [[ "$html" == *"AIMS Cameroon Assistant"* ]] || fail "web UI not reachable through the load balancer at $APP_URL"
  ok "web UI serves $APP_URL"

  body="$(remote "$API_HOST" api-health)" || fail "api health check failed"
  chunks="$(echo "$body" | json_get "d['chunks']")"
  (( chunks > 0 )) || fail "knowledge base is empty"
  ok "api healthy, $chunks chunks indexed"

  body="$(curl -sS -m 90 -w '\n%{http_code}' -X POST "$APP_URL/api/ask" \
    -H 'content-type: application/json' -d '{"question":"Where is AIMS Cameroon located?"}')"
  code="$(echo "$body" | tail -1)"
  body="$(echo "$body" | sed '$d')"
  [[ "$code" == "200" ]] || fail "grounded question returned HTTP $code: $body"
  answer="$(echo "$body" | json_get "d['answer']")"
  sources="$(echo "$body" | json_get "len(d['trace'])")"
  [[ -n "$answer" && "$answer" != "$REFUSAL"* && "$sources" -gt 0 ]] || fail "expected a grounded answer, got: $answer"
  ok "grounded answer with $sources sources: ${answer:0:70}..."

  body="$(curl -sS -m 90 -X POST "$APP_URL/api/ask" \
    -H 'content-type: application/json' -d '{"question":"What is the capital of Japan?"}')"
  answer="$(echo "$body" | json_get "d.get('answer', '')")"
  [[ "$answer" == "$REFUSAL"* ]] || fail "off-topic question was not refused: $answer"
  ok "off-topic question refused"

  if [[ "${CHECK_EXPOSURE:-1}" == "1" ]]; then
    local target
    for target in "$WEB_HOST:80" "$API_HOST:8000" "$DB_HOST:5432"; do
      if timeout 5 bash -c "</dev/tcp/${target%:*}/${target#*:}" 2>/dev/null; then
        fail "$target is reachable from the internet; only the load balancer should be"
      fi
    done
    ok "web:80, api:8000 and db:5432 are closed to the internet"
  fi
}

CURRENT_STAGE=""
# Stages run directly under `set -e` (not inside `if`, which would disable it); failures land here.
on_exit() {
  local status=$?
  [[ $status -ne 0 && -n "$CURRENT_STAGE" ]] || return 0
  echo "${red}${bold}✗ $CURRENT_STAGE failed${reset}"
  if (( DEPLOYED_WEB )); then
    echo "${yellow}Rolling back web...${reset}"
    remote "$WEB_HOST" rollback web || true
  fi
  if (( DEPLOYED_API )); then
    echo "${yellow}Rolling back api...${reset}"
    remote "$API_HOST" rollback api || true
  fi
}
trap on_exit EXIT

log "Stages: ${STAGES[*]}"
for s in "${STAGES[@]}"; do
  echo
  log "▶ $s"
  start=$SECONDS
  CURRENT_STAGE="$s"
  "stage_$s"
  log "✔ $s ($((SECONDS - start))s)"
done
CURRENT_STAGE=""

echo
if [[ " ${STAGES[*]} " == *" deploy "* || " ${STAGES[*]} " == *" smoke "* ]]; then
  log "${green}Release $(release_tag) is live at $APP_URL${reset}"
fi
