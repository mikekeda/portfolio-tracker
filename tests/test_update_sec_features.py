"""Companyfacts refresh decisions."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import update_sec_features as module


@pytest.mark.parametrize("cik, expected", [("1805385", False), ("2043699", True)])
def test_corrected_cik_refetches_fresh_companyfacts(cik, expected):
    session = Mock()
    session.get.return_value = SimpleNamespace(cik="0001805385", fetched_at=datetime.now())
    session.execute.return_value.scalar.return_value = None
    assert module._should_fetch(session, 4509, cik, force=False) is expected
