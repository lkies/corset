from corset.solver import Lens, Region, lens_combinations


def _regions_repr(setup):
    """Represent the generated setup as tuple-of-tuples of focal lengths for easy assertions."""
    return tuple(tuple(lens.focal_length for lens in region) for region in setup)


class TestLensCombinations:
    def test_empty_regions_min_zero_yields_empty_setup(self):
        out = list(lens_combinations([], [], 0, 0))
        # Should yield a single empty setup (no regions)
        assert out == [[]]

    def test_empty_regions_with_positive_min_yields_nothing(self):
        out = list(lens_combinations([], [], 1, 2))
        assert out == []

    def test_single_region_uses_base_selection_when_selection_empty(self):
        base = [Lens(1.0, 0.0, 0.0), Lens(2.0, 0.0, 0.0)]
        reg = Region(left=0.0, right=1.0, min_elements=1, max_elements=2, selection=[])
        out = list(lens_combinations([reg], base, 1, 2))

        # Expect combinations_with_replacement of base of sizes 1 and 2
        expected = {((1.0,),), ((2.0,),), ((1.0, 1.0),), ((1.0, 2.0),), ((2.0, 2.0),)}
        got = set(_regions_repr(s) for s in out)
        assert got == expected

    def test_multiple_regions_and_global_limits(self):
        A = Lens(10.0, 0.0, 0.0)
        B = Lens(20.0, 0.0, 0.0)
        C = Lens(30.0, 0.0, 0.0)

        # First region: exactly one element chosen from {A,B}
        r1 = Region(left=0.0, right=1.0, min_elements=1, max_elements=1, selection=[A, B])
        # Second region: 0..2 elements chosen from {C}
        r2 = Region(left=1.0, right=2.0, min_elements=0, max_elements=2, selection=[C])

        out = list(lens_combinations([r1, r2], [], 1, 3))

        # Build expected: r1 must be (10,) or (20,); r2 can be (), (30,), (30,30)
        expected = set()
        for first in ((10.0,), (20.0,)):
            for second in ((), (30.0,), (30.0, 30.0)):
                expected.add((first, second))

        got = set(_regions_repr(s) for s in out)
        assert got == expected

    def test_region_min_greater_than_global_max_yields_nothing(self):
        A = Lens(1.0, 0.0, 0.0)
        r = Region(left=0.0, right=1.0, min_elements=3, max_elements=3, selection=[A])
        out = list(lens_combinations([r], [], 0, 2))
        assert out == []
