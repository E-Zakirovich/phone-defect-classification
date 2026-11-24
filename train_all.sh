#!/usr/bin/env bash
#
# Train every architecture one after another, keeping one checkpoint per arch.
#
#   ./train_all.sh                       # lenet -> alexnet -> resnet, 30 epochs each
#   EPOCHS=40 ./train_all.sh             # longer run
#   ARCHES="resnet" ./train_all.sh       # just one
#   RESUME=1 ./train_all.sh              # continue from existing checkpoints
#   DEVICE=cuda ./train_all.sh           # fail fast instead of training on CPU
#
# Checkpoints : models/<arch>.pt      (best epoch only, overwritten per epoch)
# Logs        : logs/<arch>.csv
#
# Run it detached so closing the terminal does not kill the run:
#   nohup ./train_all.sh > logs/train_all.log 2>&1 &
#   tail -f logs/train_all.log
#
set -uo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
ARCHES="${ARCHES:-lenet alexnet resnet}"
EXTRA="${EXTRA:---test}"
RESUME="${RESUME:-0}"

mkdir -p models logs

echo "device=$DEVICE  epochs=$EPOCHS  arches=$ARCHES  extra=$EXTRA"

# fail immediately if the requested device is not actually available, instead
# of silently training on CPU for two days
"$PY" - "$DEVICE" <<'EOF'
import sys, torch
dev = sys.argv[1]
if dev.startswith("cuda") and not torch.cuda.is_available():
    print("ERROR: cuda requested but torch.cuda.is_available() is False.")
    print("       run on CPU with DEVICE=cpu, or fix the driver first.")
    raise SystemExit(1)
print(f"device check ok: {dev} = {torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu only'}")
EOF
[ $? -ne 0 ] && exit 1

started_all=$(date +%s)
failed=()

for arch in $ARCHES; do
    echo
    echo "############################################################"
    echo "#  $arch   ($(date '+%H:%M:%S'))"
    echo "############################################################"

    resume_arg=""
    if [ "$RESUME" = "1" ] && [ -f "models/$arch.state.pt" ]; then
        resume_arg="--resume models/$arch.state.pt"
        echo "#  resuming from models/$arch.state.pt"
    fi

    start=$(date +%s)
    # shellcheck disable=SC2086
    if "$PY" main.py --arch "$arch" --device "$DEVICE" --epochs "$EPOCHS" $resume_arg $EXTRA; then
        echo "#  $arch finished in $(( $(date +%s) - start ))s"
    else
        echo "#  $arch FAILED after $(( $(date +%s) - start ))s"
        failed+=("$arch")
    fi
done

echo
echo "================ summary ($(( ($(date +%s) - started_all) / 60 )) min total) ================"
ls -lh models/*.pt 2>/dev/null || echo "no checkpoints written"
echo
for f in logs/*.csv; do
    [ -f "$f" ] && printf '%-28s %s epochs\n' "$f" "$(( $(wc -l < "$f") - 1 ))"
done

if [ ${#failed[@]} -gt 0 ]; then
    echo
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo
echo "all architectures done."
