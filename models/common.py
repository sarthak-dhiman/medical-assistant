# Dummy stubs for classes referenced by the checkpoint's pickled objects.
# These are minimal placeholders to allow unpickling to import the names.
class Dummy: pass

_names = [
    'Concatr','Concatr3d','Concatr5','ConcatrCL','ConcatrQ','Concatrl','ConcatroC','ConcatrtW',
    'Convq','Convr','Convr_3','Convr3','Convr3L','Convr3T','Convr8I','Convr96','ConvrA','ConvrD9',
    'ConvrE','ConvrF','ConvrG','ConvrIc','ConvrIj','ConvrK','ConvrL','ConvrNH','ConvrNP','ConvrNX',
    'ConvrP','ConvrSE','ConvrT','ConvrTf','ConvrV','ConvrW','ConvrY','ConvrYD','ConvrYK','ConvrYS',
    'Convr_5','Convr_C','Convra','Convrd','Convrf','Convrg','Convrj8','Convrk','Convrm','ConvroZ',
    'Convrob','Convroi','Convrq','Convrr','Convrq','ConvrtG','ConvrtO','Convrx','Convrz','ConvrzU',
    'Convrze','DownCr','DownCrL','DownCrdW','ReOrgqUX','SPPCSPCr','Shortcutr','ShortcutrN'
]

for _n in _names:
    # sanitize name to valid identifier
    name = ''.join(c if (c.isalnum() or c=='_') else '_' for c in _n)
    if not name[0].isalpha() and name[0] != '_':
        name = '_' + name
    if name in globals():
        continue
    globals()[name] = type(name, (Dummy,), {})

__all__ = list(n for n in globals() if n[0].isalpha())

def __getattr__(name):
    # create a dummy class on demand for any missing symbol referenced during unpickling
    if name in globals():
        return globals()[name]
    cls = type(name, (Dummy,), {})
    globals()[name] = cls
    return cls
