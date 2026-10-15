#!/usr/bin/env bash
# Server-side half of the pipeline, streamed over ssh by deploy.sh:
#   ssh host 'bash -s -- <command> [args]' < deploy/remote.sh
# Each server keeps its tier in ~/aims-rag: compose.yml, .env (secrets),
# release.env (live IMAGE_TAG) and release.prev.env (rollback target).
set -euo pipefail

APP="$HOME/aims-rag"
CMD="${1:?command required}"
shift

dc() {
  sudo docker compose --project-directory "$APP" -f "$APP/compose.yml" --env-file "$APP/release.env" "$@"
}

write_release() {
  printf 'IMAGE_TAG=%s\nHOST_UID=%s\nHOST_GID=%s\n' "$1" "$(id -u)" "$(id -g)" > "$APP/release.env"
}

case "$CMD" in
  setup)
    if command -v cloud-init >/dev/null 2>&1; then
      sudo cloud-init status --wait >/dev/null || true
    fi
    if ! command -v docker >/dev/null 2>&1; then
      curl -fsSL https://get.docker.com | sudo sh
    fi
    mkdir -p "$APP"
    [[ -f "$APP/release.env" ]] || write_release none
    free_gb="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')"
    if (( free_gb < 3 )); then
      echo "only ${free_gb}G disk free" >&2
      exit 1
    fi
    echo "$(hostname): docker $(sudo docker version --format '{{.Server.Version}}'), ${free_gb}G free"
    ;;

  has-image)
    sudo docker image inspect "$1" >/dev/null 2>&1
    ;;

  db-up)
    dc up -d --wait db
    ;;

  release)
    service="$1"
    tag="$2"
    current="$(sed -n 's/^IMAGE_TAG=//p' "$APP/release.env")"
    if [[ "$current" != "$tag" && "$current" != "none" ]]; then
      cp "$APP/release.env" "$APP/release.prev.env"
    fi
    write_release "$tag"
    if [[ "$service" == "api" ]]; then
      # Sync the knowledge base with the new code before switching traffic: embeds only new/edited chunks.
      dc --profile tools run --rm -T indexer
    fi
    dc up -d --wait "$service"
    ;;

  rollback)
    service="$1"
    if [[ ! -f "$APP/release.prev.env" ]]; then
      echo "$service: no previous release to roll back to" >&2
      exit 1
    fi
    mv "$APP/release.prev.env" "$APP/release.env"
    dc up -d --wait "$service"
    echo "$service rolled back to $(sed -n 's/^IMAGE_TAG=//p' "$APP/release.env")"
    ;;

  api-health)
    curl -fsS http://localhost:8000/health
    ;;

  prune)
    repo="$1"
    keep=()
    for f in "$APP/release.env" "$APP/release.prev.env"; do
      [[ -f "$f" ]] && keep+=("$(sed -n 's/^IMAGE_TAG=//p' "$f")")
    done
    for tag in $(sudo docker images "$repo" --format '{{.Tag}}'); do
      if [[ " ${keep[*]} " != *" $tag "* ]]; then
        sudo docker rmi "$repo:$tag" >/dev/null 2>&1 || true
      fi
    done
    sudo docker image prune -f >/dev/null
    ;;

  *)
    echo "unknown command: $CMD" >&2
    exit 1
    ;;
esac
