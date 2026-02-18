#!/bin/bash
mkdir -p /tmp/fpocket_work && cd /tmp/fpocket_work
rm -rf '4MLH_out' '4MLH.pdb' 2>/dev/null
cp '/mnt/e/newyear/research_plan/allosteric/data/pdb_files/4MLH.pdb' . && fpocket -f '4MLH.pdb' >/dev/null 2>&1
echo '===INFO_START==='
cat '4MLH_out/4MLH_info.txt' 2>/dev/null
echo '===INFO_END==='
for f in 4MLH_out/pockets/pocket*_atm.pdb; do
  [ -f "$f" ] || continue
  num=$(echo "$f" | grep -oP 'pocket\K[0-9]+')
  echo "===POCKET_${num}_START==="
  grep ^ATOM "$f" 2>/dev/null
  echo "===POCKET_${num}_END==="
done
rm -rf '4MLH_out' '4MLH.pdb' 2>/dev/null
