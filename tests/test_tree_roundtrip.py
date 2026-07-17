from finrag.parse.tree import DocumentTree


def test_tree_json_roundtrip(tree, tmp_path):
    p = tmp_path / "t.json"
    tree.to_json(p)
    back = DocumentTree.from_json(p)
    assert back.doc_name == tree.doc_name
    assert back.n_blocks() == tree.n_blocks()
    paths = [" > ".join(path) for _, path in back.walk()]
    assert any("Item 7" in p and "Non-GAAP" in p for p in paths)
