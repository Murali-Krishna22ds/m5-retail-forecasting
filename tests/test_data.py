import numpy as np


def test_subset_shape_and_hierarchy(data):
    bt, at, S, h = data
    assert h.sku.nunique() == 400 and h.store_id.nunique() == 10 and h.state_id.nunique() == 3
    assert h.item_id.nunique() == 40 and set(h.cat_id) == {"FOODS", "HOUSEHOLD", "HOBBIES"}
    assert S.shape == (1941, 414)
    assert not bt.duplicated(["sku", "d"]).any()


def test_aggregation_consistency(data):
    bt, at, S, h = data
    bottoms = S[list(h.sku)]
    assert np.allclose(S["Total"], bottoms.sum(axis=1))
    for s, g in h.groupby("state_id"):
        assert np.allclose(S[s], bottoms[list(g.sku)].sum(axis=1))
    for s, g in h.groupby("store_id"):
        assert np.allclose(S[s], bottoms[list(g.sku)].sum(axis=1))


def test_sales_non_negative_integers(data):
    bt, *_ = data
    assert (bt.y >= 0).all() and np.allclose(bt.y, bt.y.round())


def test_no_price_means_no_sales_mostly(data):
    """M5 convention: no price record => item not on sale => zero sales."""
    bt, *_ = data
    assert bt.loc[bt.sell_price.isna(), "y"].sum() == 0
