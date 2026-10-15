#!/usr/bin/env bash
# Run Terraform from its Docker image (nothing installed on the host).
#   ./infra/tf.sh init | plan | apply | output | destroy
set -euo pipefail
cd "$(dirname "$0")"

PUB_KEY_FILE="${SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/aims-rag.pub}"
if [[ ! -f "$PUB_KEY_FILE" ]]; then
  echo "SSH public key $PUB_KEY_FILE not found. Create one with:" >&2
  echo "  ssh-keygen -t ed25519 -f ~/.ssh/aims-rag -C aims-rag" >&2
  exit 1
fi

ENV_ARGS=()
if [[ -f aws.env ]]; then
  ENV_ARGS=(--env-file aws.env)
elif [[ "${1:-}" != "output" ]]; then
  echo "infra/aws.env not found. Copy infra/aws.env.example and add your access key." >&2
  exit 1
fi

TTY_ARGS=()
[[ -t 0 && -t 1 ]] && TTY_ARGS=(-it)

exec docker run --rm "${TTY_ARGS[@]}" \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e TF_IN_AUTOMATION=1 \
  -e TF_VAR_ssh_public_key="$(cat "$PUB_KEY_FILE")" \
  "${ENV_ARGS[@]}" \
  -v "$PWD":/workspace -w /workspace \
  hashicorp/terraform:1.9.8 "$@"
