
__version__ = "0.1"


def resolve_ambiguous_requirements(options):
    """
    Since canopy has different names for certain packages in the package
    manager, we will check to see whether canopy distribution is being
    used or not and adjust the requirments accordingly.
    List both options for any ambiguous packages here as tuples
    (canopy name, python name, version requirement)
    """
    import sys
    exe_path = sys.prefix
    use_canopy = 'canopy' in exe_path.lower()
    if use_canopy:
        canopy_names = [option[0] + option[2] for option in options]
    else:
        canopy_names = [option[1] + option[2] for option in options]
    return canopy_names


_non_ambiguous_requirements = [
    "apptools>=4.2.0",
    "numpy>=1.6",
    "traits>=4.4",
    "enable>4.2",
    "chaco>=4.4",
    "fiona>=1.0.2",
    "scimath>=4.1.2",
    "shapely>=1.2.17",
    "sdi",
    "scipy",
]

_ambiguous_requirements = [
    ('pytables', 'tables', '>=2.4.0'),
    ('scikits.image', 'skimage', ''),
    ('scikit_learn', 'sklearn', '')
    ]
_resolved = resolve_ambiguous_requirements(_ambiguous_requirements)

__requires__ = _non_ambiguous_requirements + _resolved

