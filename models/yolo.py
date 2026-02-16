"""Minimal yolo stubs used by the checkpoint.

We add a `Model` class because the checkpoint references `models.yolo.Model`.
Additional classes are added as simple placeholders.
"""
class Dummy: pass

class Model(Dummy):
    def __init__(self, *a, **k):
        pass

_names = ['IAuxDetectrnn']
for _n in _names:
    name = ''.join(c if (c.isalnum() or c=='_') else '_' for c in _n)
    if not name[0].isalpha() and name[0] != '_':
        name = '_' + name
    if name not in globals():
        globals()[name] = type(name, (Dummy,), {})

__all__ = ['Model'] + _names

def __getattr__(name):
    # create a dummy class on demand for missing symbols
    if name in globals():
        return globals()[name]
    cls = type(name, (Dummy,), {})
    globals()[name] = cls
    return cls
