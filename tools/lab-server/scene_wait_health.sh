#!/usr/bin/env bash
# Poll the policy server health endpoint until it answers ok (or we give up).
for i in $(seq 1 90); do
  if curl -s -m 5 http://127.0.0.1:8001/health 2>/dev/null | grep -q ok; then
    echo "HEALTHY after $((i * 20)) s"
    curl -s http://127.0.0.1:8001/health; echo
    exit 0
  fi
  sleep 20
done
echo "TIMEOUT waiting for server"
tail -30 ~/server.log
exit 1
