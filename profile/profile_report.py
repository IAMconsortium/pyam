import sqlite3
import pandas as pd


def main():
    db = sqlite3.connect(".pymon")

    print(
        pd.read_sql_query(
            "SELECT item_path, item_variant, total_time, cpu_usage, mem_usage "
            "FROM test_metrics WHERE session_h = ("
            "  SELECT session_h FROM test_sessions "
            "  WHERE run_date = (SELECT MAX(run_date) FROM test_sessions)"
            ")",
            db,
        )
    )


if __name__ == "__main__":
    main()
