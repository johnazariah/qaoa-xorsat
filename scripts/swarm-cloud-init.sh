#!/bin/bash
# Cloud-init script for QAOA swarm VM
# Runs sequential swarm optimizer for all 5 hard (k,D) pairs

set -euo pipefail
exec > /var/log/qaoa-swarm-init.log 2>&1

echo "$(date -u): Starting QAOA swarm VM setup"

# Install Julia
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12
export PATH="$HOME/.juliaup/bin:$PATH"

# Clone repo
cd /home/azureuser
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "$(date -u): Setup complete, launching swarm"

# Run sequential swarm -- one pair at a time, all cores
for pair in "5 7" "5 8" "6 7" "6 8" "7 8"; do
    k=$(echo $pair | cut -d" " -f1)
    D=$(echo $pair | cut -d" " -f2)
    logfile="/home/azureuser/swarm-k${k}d${D}.log"
    echo "$(date -u): Starting (k=$k, D=$D)" | tee -a /home/azureuser/swarm-master.log
    julia --project=. -t auto scripts/swarm_chain.jl $k $D 12 100 10 20 42 > "$logfile" 2>&1
    echo "$(date -u): Finished (k=$k, D=$D)" | tee -a /home/azureuser/swarm-master.log
    tail -20 "$logfile" >> /home/azureuser/swarm-master.log
done

echo "$(date -u): ALL PAIRS COMPLETE" | tee -a /home/azureuser/swarm-master.log

# Push results to a branch
cd /home/azureuser/qaoa-xorsat
git checkout -b swarm-vm-results
for f in /home/azureuser/swarm-k*.log /home/azureuser/swarm-master.log; do
    cp "$f" results/ 2>/dev/null || true
done
git add -A results/
git commit -m "swarm: VM results for high (k,D) pairs" || true
git push origin swarm-vm-results || true

echo "$(date -u): Done and pushed"
