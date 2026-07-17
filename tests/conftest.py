import pytest

from finrag.parse.tree import Block, DocumentTree, Section

TABLE_MD = (
    "| Metric | FY2022 | FY2023 |\n"
    "|---|---|---|\n"
    "| Net income | 1,000 | 1,200 |\n"
    "| Adjusted EBITDA | 1,800 | 2,018 |"
)


@pytest.fixture
def tree() -> DocumentTree:
    prose = "The company reported strong results. " * 40  # long enough to split
    item7 = Section(
        id="s1",
        title="Item 7. Management's Discussion and Analysis",
        level=1,
        blocks=[Block(id="b1", kind="paragraph", text=prose)],
        children=[
            Section(
                id="s2",
                title="Non-GAAP Reconciliation",
                level=2,
                blocks=[
                    Block(id="b2", kind="paragraph", text="Reconciliation of net income."),
                    Block(id="b3", kind="table", text=TABLE_MD, caption="Adjusted EBITDA"),
                ],
            )
        ],
    )
    item8 = Section(
        id="s3",
        title="Item 8. Financial Statements",
        level=1,
        blocks=[Block(id="b4", kind="paragraph", text="Consolidated statements follow.")],
    )
    return DocumentTree(doc_name="TEST_2023_10K", company="TestCo", sections=[item7, item8])
