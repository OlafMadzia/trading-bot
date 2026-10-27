from bot import live


def test_append_csv_merges_new_columns(tmp_path):
    path = str(tmp_path / "t.csv")
    live._append_csv(path, {"a": 1, "b": 2})
    live._append_csv(path, {"a": 3, "b": 4, "c": 5})  # a new column appears later
    rows = live._read_csv(path)
    assert [r["a"] for r in rows] == ["1", "3"]
    assert rows[0]["c"] == "" and rows[1]["c"] == "5"


def test_upsert_daily_keeps_one_row_per_date(tmp_path):
    path = str(tmp_path / "e.csv")
    live._upsert_daily(path, {"date": "2026-01-01", "v": 1})
    live._upsert_daily(path, {"date": "2026-01-02", "v": 2})
    live._upsert_daily(path, {"date": "2026-01-02", "v": 3})  # re-run same day
    rows = live._read_csv(path)
    assert [(r["date"], r["v"]) for r in rows] == [("2026-01-01", "1"), ("2026-01-02", "3")]


def test_every_tradeable_asset_has_a_trading212_ticker():
    from bot.data import CASH, EXTRA, UNIVERSE
    assert set(UNIVERSE + EXTRA + [CASH]) <= set(live.MAP)
