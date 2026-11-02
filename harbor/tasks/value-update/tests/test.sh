#!/bin/sh

mkdir -p /logs/verifier

if [ "$(cat /workspace/app.py)" = "VALUE = 2" ]; then
    echo "1" > /logs/verifier/reward.txt
else
    echo "0" > /logs/verifier/reward.txt
fi
