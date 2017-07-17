import pytest
from pytest import raises
from jzl.keras.layers import Force

def test_force_init():
    x = Force(1000)
    assert isinstance(x, Force)
    assert x.units==1000
    assert x.readouts==1
    assert hasattr(x,'input_spec')
    assert hasattr(x,'state_spec')
    
