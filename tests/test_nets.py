import numpy as np

from haven.nets import MLP, Prototypes, QualityCoder, RunningStat, cosine, softmax


def test_mlp_learns_a_function():
    rng = np.random.default_rng(0)
    net = MLP([2, 16, 1], rng, lr=0.05)
    xs = rng.uniform(-1, 1, (400, 2))
    ys = np.sin(2 * xs[:, 0]) * xs[:, 1]
    before = np.mean([(net(x)[0] - y) ** 2 for x, y in zip(xs, ys, strict=True)])
    for _ in range(30):
        for x, y in zip(xs, ys, strict=True):
            net.learn(x, np.array([y]))
    after = np.mean([(net(x)[0] - y) ** 2 for x, y in zip(xs, ys, strict=True)])
    assert after < before / 4


def test_quality_code_is_sparse_and_smooth():
    rng = np.random.default_rng(0)
    coder = QualityCoder(3, 16, rng, width=0.16, active=3)
    for _ in range(2000):
        coder.learn(rng.uniform(0, 1, 3), 0.05)
    a = coder.tuning(np.array([0.9, 0.2, 0.2]))
    near = coder.tuning(np.array([0.88, 0.22, 0.2]))
    far = coder.tuning(np.array([0.1, 0.8, 0.3]))
    assert (a > 0).sum() <= 3
    assert np.linalg.norm(a - near) < np.linalg.norm(a - far)


def test_new_kinds_need_confirming():
    kinds = Prototypes(2, capacity=4, radius=0.3, confirm=3)
    assert kinds.assign(np.array([1.0, 0.0])) == -1  # a first glimpse is only a candidate
    assert kinds.assign(np.array([1.0, 0.05])) == -1
    assert kinds.assign(np.array([0.98, 0.0])) == 0  # seen enough: a kind of its own
    assert kinds.assign(np.array([1.0, 0.02])) == 0
    assert kinds.assign(np.array([0.0, 1.0]), may_create=False) == -1


def test_small_helpers():
    stat = RunningStat(0.5, 0.0, 1.0)
    for _ in range(20):
        stat.update(2.0)
    assert abs(stat.mean - 2.0) < 1e-3
    assert np.isclose(softmax(np.array([1.0, 1.0])).sum(), 1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
