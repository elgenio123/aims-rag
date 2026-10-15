#!/bin/bash
# cloud-init: runs once at first boot of the ${role} server.
set -euxo pipefail

hostnamectl set-hostname "aims-rag-${role}"

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
usermod -aG docker ubuntu
systemctl enable --now docker

if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

mkdir -p /home/ubuntu/aims-rag
chown ubuntu:ubuntu /home/ubuntu/aims-rag
