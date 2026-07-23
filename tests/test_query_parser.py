from query_parser import QueryParser

def test_query_parser():
    parser = QueryParser()
    usages = parser.parse('SELECT * FROM users WHERE age > 18')
    assert len(usages) == 1
    assert usages[0].table == 'users'
    assert usages[0].column == 'age'
    assert usages[0].clause == 'WHERE'
    assert usages[0].operator == 'GT'
    
    usages2 = parser.parse('SELECT * FROM a JOIN b ON a.b_id = b.id ORDER BY a.val GROUP BY b.name')
    clauses = {u.clause: u for u in usages2}
    assert 'JOIN' in clauses
    assert 'ORDER_BY' in clauses
    assert 'GROUP_BY' in clauses
