import pytest

from sandbox import Sandbox


@pytest.fixture
def sandbox(csv_path):
    sb = Sandbox(csv_path, csv_path.parent, timeout=10)
    yield sb
    sb.close()


def test_prints_value_of_last_expression(sandbox):
    assert sandbox.run("df.groupby('city').sales.sum().to_dict()") == (True, "{'Lyon': 5, 'Paris': 17}\n")


def test_variables_persist_between_calls(sandbox):
    sandbox.run("total = df.sales.sum()")
    assert sandbox.run("print(total * 2)") == (True, "44\n")


def test_error_points_at_the_cell_line_not_pandas_internals(sandbox):
    ok, output = sandbox.run("x = 1\ndf['missing']")
    assert not ok
    assert "line 2: df['missing']" in output
    assert "KeyError" in output
    assert "site-packages" not in output


def test_long_output_is_truncated(sandbox):
    ok, output = sandbox.run("print('x' * 50_000)")
    assert ok and "characters truncated" in output and len(output) < 7000


def test_chart_is_saved(sandbox):
    ok, _ = sandbox.chart("df.groupby('city').sales.sum().plot.bar(title='Sales')", "sales.png")
    assert ok and (sandbox.out_dir / "sales.png").stat().st_size > 0


def test_chart_that_draws_nothing_is_an_error(sandbox):
    ok, output = sandbox.chart("x = 1", "empty.png")
    assert not ok and "didn't draw anything" in output


def test_timeout_restarts_the_session(csv_path):
    sb = Sandbox(csv_path, csv_path.parent, timeout=1)
    try:
        sb.run("leftover = 1")
        ok, output = sb.run("while True: pass")
        assert not ok and "Timed out" in output
        assert sb.run("len(df)") == (True, "3\n")  # df is reloaded
        ok, output = sb.run("leftover")
        assert not ok and "NameError" in output  # everything else is gone
    finally:
        sb.close()
