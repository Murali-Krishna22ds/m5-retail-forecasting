import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app"))
import app as A


def test_layout_builds_and_serves():
    c = A.app.server.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/_dash-layout").status_code == 200


def test_forecast_figure_all_levels():
    for lvl in ["Total", "State", "Store", "Item-Store"]:
        node = A.nodes_for(lvl)[0]
        fig = A.forecast_figure(node, ["lgbm_mint", "prophet"], "80", 112)
        assert len(fig.data) >= 4
        m = A.node_metrics(node, ["lgbm_mint", "prophet"]); assert {"WAPE", "RMSSE"} <= set(m.columns)


def test_accuracy_and_reconciliation_figures():
    for met in ["WAPE", "RMSSE", "SPL", "Cov80", "Cov95"]:
        assert len(A.accuracy_figure(met, "Item-Store").data) == 1
    assert len(A.reconciliation_figure("TX").data) == 2


def test_order_plan_respects_constraints_and_slider_effects():
    res, k = A.plan_store("WI_2", 2, 1.05, 1.5, 1.0, 0)
    assert k["budget_used"] <= k["budget"] + 1e-6 and k["cap_used"] <= k["cap"] + 1e-6
    assert (res["Order qty"] >= 0).all() and k["status"] in ("OPTIMAL", "FEASIBLE")
    tight, kt = A.plan_store("WI_2", 2, 0.7, 1.5, 1.0, 0)
    loose, kl = A.plan_store("WI_2", 2, 1.5, 1.5, 1.0, 0)
    assert tight["Order $"].sum() <= res["Order $"].sum() <= loose["Order $"].sum() + 1e-6
    assert kl["exp_fill"] >= kt["exp_fill"]
    # higher stockout penalty -> order more (shelf capacity relaxed so it cannot be the binding constraint)
    lo_pen, _ = A.plan_store("WI_2", 2, 3.0, 5.0, 1.0, 0); hi, _ = A.plan_store("WI_2", 2, 3.0, 5.0, 3.0, 0)
    assert hi["Order qty"].sum() >= lo_pen["Order qty"].sum()
    stocked, _ = A.plan_store("WI_2", 2, 1.5, 1.5, 1.0, 5)       # on-hand stock -> order less
    assert stocked["Order qty"].sum() < loose["Order qty"].sum()
    assert len(A.order_figure(res).data) == 3 and len(A.policy_figure().data) == 1


def test_every_model_at_every_level_does_not_crash():
    models = A.store_data()["models"]
    for lvl in ["Total", "State", "Store", "Item-Store"]:
        node = A.nodes_for(lvl)[0]
        for m in models:
            for band in ("80", "95", "none"):
                A.forecast_figure(node, [m], band, 56)
            assert len(A.node_metrics(node, [m])) == 1
