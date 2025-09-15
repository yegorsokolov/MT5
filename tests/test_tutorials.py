import doctest
import re
from pathlib import Path

import pytest


TUTORIALS = Path("docs/tutorials")


@pytest.mark.parametrize("path", TUTORIALS.glob("*.md"))
def test_tutorial_doctests(path: Path):
    text = path.read_text()
    blocks = re.findall(r"```{doctest}\n(.*?)```", text, re.DOTALL)
    parser = doctest.DocTestParser()
    for block in blocks:
        runner = doctest.DocTestRunner()
        test = parser.get_doctest(block, {}, str(path), str(path), 0)
        result = runner.run(test)
        assert result.failed == 0
