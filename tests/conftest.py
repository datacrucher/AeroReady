import pytest

from aeroready import db, model, seed


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    return model.load_or_train(path=tmp_path_factory.mktemp("model") / "risk_model.pkl")


@pytest.fixture
def fresh_db(tmp_path, engine):
    """A freshly seeded, isolated database for each test."""
    old = db.get_db_path()
    db.set_db_path(tmp_path / "test.db")
    seed.seed(engine)
    yield
    db.set_db_path(old)
