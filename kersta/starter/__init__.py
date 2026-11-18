"""Starter detectors, one per lane: the score every miner has to beat.

Each lane folder holds:
    package/model.py           the detector (torch); passes the sandbox policy
    package/model_config.yaml  its config
    train.py                   training, with Kersta's own preprocessing and damage tool

``kersta miner init --lane <lane>`` copies package/ into a submission folder.
The starter weights the subnet owner trains and publishes are the baseline on
the leaderboard; this code reproduces them.
"""
