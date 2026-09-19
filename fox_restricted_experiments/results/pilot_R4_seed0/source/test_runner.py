"""Integration checks for paired training, lag evaluation and saved artifacts."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import Config, Model
from runner import evaluate, run_suite


def rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_four_arms_from_identical_initialization(tmp_path):
    torch.set_num_threads(1)
    cfg = Config(n=4,d=64,R=4,steps=7,device="cpu",pair_batch=16)
    result = Path(run_suite(cfg,tmp_path/"run",seeds=(3,),eval_lags=(1,4,6,7),
                           prefixes=(0,8),log_every=5,progress=False))
    runs = rows(result/"runs.csv")
    assert len(runs)==4
    assert len({r["initial_hash"] for r in runs})==1
    assert all(r["status"]=="finite_budget_complete" for r in runs)
    assert all(r["acquired_positive_gaps"]=="True" for r in runs)
    hist = rows(result/"history.csv")
    fixed = [float(r["h"]) for r in hist if r["gate_mode"]=="retrieval_frozen"]
    assert max(fixed)==min(fixed)
    assert np.isclose(fixed[0],cfg.h0)
    assert all(a["passed"] for a in json.loads((result/"native_gradient_audits.json").read_text()))
    prob = rows(result/"lag_probabilities.csv")
    assert {int(r["lag"]) for r in prob}=={1,4,6,7}
    assert all(0<=float(r["min_probability"])<=float(r["mean_probability"])+1e-15<=1+1e-15 for r in prob)
    for mode in ("learned","retrieval_frozen"):
        ck = torch.load(result/"checkpoints"/f"seed3_{mode}_adam.pt",weights_only=False)
        assert ck["optimizer"]["t"]==cfg.steps  # history retained through acquisition
    with pytest.raises(FileExistsError):
        run_suite(cfg,result,seeds=(3,),progress=False)


def test_bound_below_finite_prefix_probability():
    cfg = Config(n=3,d=8,R=2,device="cpu")
    model = Model(cfg)
    # This is a mathematical evaluation fixture, never a training initialization.
    model.Q.zero_(); model.K.zero_()
    model.Q[:,:3] = torch.eye(3,dtype=torch.float64)*.5
    model.K.copy_(model.Q)
    model.theta[0:2] = 4.
    model.theta[-1] = 3.
    result = evaluate(model,dict(seed=0,gate_mode="learned",optimizer="sgd"),0,0.,range(1,7),(0,1,16,128))
    assert all(r["certified_probability_lower_bound"]<=r["min_probability"]+1e-14 for r in result)
