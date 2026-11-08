from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_interactions():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not app.exception
    assert app.metric[0].label == "Artifact suspicion"
    app.selectbox[0].select("single_jpeg").run()
    assert not app.exception
    app.slider[0].set_value(80).run()
    assert not app.exception
