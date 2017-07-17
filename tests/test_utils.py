import pytest
from pytest import raises

from jzl.utils import metrics
import jzl.utils.spectra as spectra

def test_existence():
    assert callable(spectra.mtspec)
    assert callable(spectra.rolling_window)


def test_si():
    with pytest.raises(TypeError) as excinfo:
        # should take 1 argument
        s = metrics.si()
