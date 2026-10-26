#!/usr/bin/env bash
# Container entrypoint for the hohere image on RunPod: SSH, then vLLM, then stay alive.

# SSH with the account key RunPod injects as $PUBLIC_KEY.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A > /dev/null
/usr/sbin/sshd

# SSH sessions do not inherit the container env; expose the pod's variables to them.
env | grep -E '^(HF_|HOHERE_|GITHUB_TOKEN|RUNPOD_|UV_|PATH=)' | while IFS='=' read -r k v; do
  printf 'export %s=%q\n' "$k" "$v"
done > /etc/profile.d/hohere_env.sh
echo '. /etc/profile.d/hohere_env.sh' >> /root/.bashrc

/opt/hohere/scripts/post_start.sh

exec sleep infinity
