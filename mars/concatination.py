import typing as tp

from . import spin_model
from . import population


mars_items_types = tp.Union[
    tp.Sequence[spin_model.SpinSystem],
    tp.Sequence[spin_model.MultiOrientedSample],
    tp.Sequence[population.BaseContext]
]

mars_item_type = tp.Union[
    spin_model.SpinSystem,
    spin_model.MultiOrientedSample,
    population.BaseContext
]


def concat(mars_items: mars_items_types) -> mars_item_type:
    """
    Concatenate a sequence of homogeneous MarS items into a single combined item.

    This function dispatches to the appropriate concatenation routine based on the type
    of the first element in the input sequence. It supports three types of inputs:

    - A sequence of :class:`SpinSystem` -> returns a single block-diagonal :class:`SpinSystem`
    - A sequence of :class:`MultiOrientedSample` -> returns a single :class:`MultiOrientedSample`
      with a concatenated spin system and shared spectral parameters
    - A sequence of :class:`Context` (or :class:`SummedContext`) -> returns a composite
      context that computed direct sum of all contexts

    The input sequence must be non-empty and contain only one kind of item type.

    In all cases, the direct sum of the elements you want to concatenate is calculated.
    To understand the logic and interpretation of operation in each case, see

    - :func:`mars.spin_model.concat_spin_system`
    - :func:`mars.spin_model.concat_multioriented_samples`
    - :func:`mars.population.concatinations.concat_contexts`

    :param mars_items: A non-empty sequence of identical-type MARS objects.
                       Must be one of:
                       - ``Sequence[SpinSystem]``
                       - ``Sequence[MultiOrientedSample]``
                       - ``Sequence[BaseContext]``

    :return: A single MARS object of the corresponding type that represents the
             logical concatenation of all input items.

    :raises TypeError: If the item type is not supported.
    :raises ValueError: If underlying concatenation logic detects inconsistencies
                        (e.g., mismatched dtypes, devices, or spectral parameters).
    """
    ref_item = mars_items[0]
    if isinstance(ref_item, spin_model.SpinSystem):
        return spin_model.concat_spin_systems(mars_items)
    elif isinstance(ref_item, spin_model.MultiOrientedSample):
        return spin_model.concat_multioriented_samples(mars_items)
    elif isinstance(ref_item, population.BaseContext):
        return population.concat_contexts(mars_items)
    else:
        raise NotImplementedError("MarS concat works only for spin systems, samples, contexts")
