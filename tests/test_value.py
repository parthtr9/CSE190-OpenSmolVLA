"""Unit tests for the value function MLP."""

import torch

from recap.value import ValueConfig, ValueMLP


def test_forward_shape():
    model = ValueMLP(ValueConfig(state_dim=6))
    state = torch.randn(8, 6)
    out = model(state)
    assert out.shape == (8,)


def test_target_normalization_roundtrip():
    model = ValueMLP(ValueConfig(state_dim=6, normalize_target=True))
    returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
    model.fit_target_normalization(returns)
    # normalize then check zero-mean unit-std-ish
    normed = model.normalize_target(returns)
    assert abs(float(normed.mean())) < 1e-5
    assert abs(float(normed.std(unbiased=True)) - 1.0) < 1e-2


def test_can_overfit_varying_target():
    # The MLP should be able to fit a non-trivial state->return mapping.
    # (A constant target is degenerate under target-normalization: std->0, so we
    # use a target that actually varies with the state.)
    torch.manual_seed(0)
    model = ValueMLP(ValueConfig(state_dim=4, hidden_dims=[64, 64]))
    state = torch.randn(128, 4)
    target = 3.0 * state[:, 0] - 2.0 * state[:, 1] + 1.0  # linear, learnable
    model.fit_target_normalization(target)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(500):
        opt.zero_grad()
        pred = model.raw_forward(state)
        loss = torch.nn.functional.mse_loss(pred, model.normalize_target(target))
        if first_loss is None:
            first_loss = float(loss)
        loss.backward()
        opt.step()
    # Loss should drop substantially, and predictions track the target.
    assert float(loss) < 0.1 * first_loss
    with torch.no_grad():
        pred = model(state)
    # Correlation between prediction and target should be high.
    corr = torch.corrcoef(torch.stack([pred, target]))[0, 1]
    assert float(corr) > 0.95
