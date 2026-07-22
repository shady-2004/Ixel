
import sqlglot
from sqlglot import exp
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColumnUsage:
    table: Optional[str]     # may be None if we can't resolve alias -> table
    column: str
    clause: str               # "WHERE" | "JOIN" | "ORDER_BY" | "GROUP_BY"
    operator: Optional[str]   # "=", "<", ">", "LIKE", etc. (only meaningful for WHERE)


class QueryParser:
    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect

    def parse(self, sql: str) -> List[ColumnUsage]:
        """
        Parse a single SQL string, return a list of ColumnUsage events.
        Returns an empty list if the query can't be parsed (never raises) —
        malformed/unusual SQL should degrade gracefully, not crash the pipeline.
        """
        try:
            tree = sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception:
            return []

        if tree is None:
            return []

        usages: List[ColumnUsage] = []
        alias_map = self._build_alias_map(tree)
        # when a column is unqualified (no "table." prefix) and the query only
        # touches one table, we can safely assume it belongs to that table
        single_table_fallback = list(alias_map.values())[0] if len(alias_map) == 1 else None

        # WHERE clause
        where = tree.find(exp.Where)
        if where:
            usages.extend(self._extract_conditions(where.this, alias_map, clause="WHERE", fallback_table=single_table_fallback))

        # JOIN conditions
        for join in tree.find_all(exp.Join):
            on_clause = join.args.get("on")
            if on_clause:
                usages.extend(self._extract_conditions(on_clause, alias_map, clause="JOIN", fallback_table=single_table_fallback))

        # ORDER BY
        order = tree.find(exp.Order)
        if order:
            for ordered in order.find_all(exp.Ordered):
                col = ordered.this
                if isinstance(col, exp.Column):
                    usages.append(self._make_usage(col, alias_map, "ORDER_BY", None, single_table_fallback))

        # GROUP BY
        group = tree.find(exp.Group)
        if group:
            for col in group.find_all(exp.Column):
                usages.append(self._make_usage(col, alias_map, "GROUP_BY", None, single_table_fallback))

        return usages

    # ---- internals ----

    def _build_alias_map(self, tree) -> dict:
        """Map table aliases -> real table names, so 'o.user_id' resolves to 'orders'."""
        alias_map = {}
        for table_exp in tree.find_all(exp.Table):
            real_name = table_exp.name
            alias = table_exp.alias or real_name
            alias_map[alias] = real_name
        return alias_map

    def _make_usage(self, col: exp.Column, alias_map: dict, clause: str, operator: Optional[str],
                     fallback_table: Optional[str] = None) -> ColumnUsage:
        table_ref = col.table  # alias or table name as written in the query, may be ""
        resolved_table = alias_map.get(table_ref, table_ref or None) or fallback_table
        return ColumnUsage(
            table=resolved_table,
            column=col.name,
            clause=clause,
            operator=operator,
        )

    def _extract_conditions(self, node, alias_map: dict, clause: str, fallback_table: Optional[str] = None) -> List[ColumnUsage]:
        """Walk a WHERE/JOIN-ON expression tree, pulling out (column, operator) pairs."""
        usages = []
        if node is None:
            return usages

        # binary comparisons: col = val, col > val, etc.
        binary_ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Between)
        for comp in node.find_all(binary_ops):
            op_name = type(comp).__name__.upper()
            for col in comp.find_all(exp.Column):
                usages.append(self._make_usage(col, alias_map, clause, op_name, fallback_table))

        return usages


if __name__ == "__main__":
    parser = QueryParser()

    test_queries = [
        "SELECT * FROM orders WHERE user_id = ? AND status = 'shipped'",
        "SELECT * FROM orders o JOIN users u ON o.user_id = u.id ORDER BY o.created_at",
        "SELECT category, COUNT(*) FROM products GROUP BY category",
        "SELECT * FROM orders WHERE amount > 100 AND amount < 500",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        for usage in parser.parse(q):
            print(f"  {usage}")
