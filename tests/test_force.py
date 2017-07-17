import pytest
from pytest import raises
from jzl.keras.layers import Force


def test_force_init():
    x = Force(1000)
    assert isinstance(x, Force)
    assert x.activation==None
    assert x.recurrent_activation=='tanh'
