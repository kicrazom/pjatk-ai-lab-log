#!/bin/bash
echo "=== $(date +%H:%M:%S) ==="
echo
echo "[procesy vLLM]"
ps -ef | grep -E 'EngineCore|test_single|test_dual' | grep -v grep | awk '{print "  PID",$2,"CPU:",$7,$NF}'
echo
echo "[pobieranie HF]"
du -sh ~/.cache/huggingface/hub/ 2>/dev/null | awk '{print "  Total:",$1}'
INC=$(find ~/.cache/huggingface/hub/ -name '*.incomplete' 2>/dev/null)
if [ -n "$INC" ]; then
    ls -lh $INC 2>/dev/null | awk '{print "  Incomplete:",$5,"  modified:",$6,$7,$8}'
else
    echo "  Incomplete: none (download complete)"
fi
echo
echo "[GPU 0]"
rocm-smi -d 0 --showuse --showmeminfo vram 2>/dev/null | grep -E 'GPU\[0\]' | sed 's/^/  /'
echo
echo "[ostatnie 5 linii logu]"
tail -5 ~/logs/02-single-gpu-inference.log 2>/dev/null | sed 's/^/  /'
