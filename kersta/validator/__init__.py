"""The validator: read commitments at the deadline, fetch packages, run each
model in the sandbox on the competition's test set, score, set weights, report.

Module map (spec: "Validator modules"):
    kersta.core.competition    the competition's blocks
    kersta.chain.commitments   commitments (format and deadline filter)
    fetcher.py                 fetcher: download, hash and size checks, cache
    kersta.sandbox             sandbox and runner
    evaluate.py, competition.py  runner glue and the scorer for a whole competition
    weights.py                 weights: submit and keep fresh
    report.py                  reporter: signed per competition reports and leaderboard
    loop.py                    the long running loop
"""
