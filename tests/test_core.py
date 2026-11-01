import numpy as np
import pytest

from corset.core import Beam, OpticalSetup, ThinLens


class TestBeam:
    def test_from_gauss_sets_focus_and_waist(self):
        beam = Beam.from_gauss(
            focus=2.0,
            waist=0.5,
            wavelength=1.0,
        )

        assert beam.focus == pytest.approx(2.0)
        assert beam.waist == pytest.approx(0.5)
        assert beam.rayleigh_range == pytest.approx(np.pi * 0.5**2)

    def test_radius_is_minimum_at_focus(self):
        beam = Beam.from_gauss(
            focus=2.0,
            waist=1.0,
            wavelength=np.pi * np.sqrt(3),
        )

        assert beam.radius(2.0) == pytest.approx(1.0)
        assert beam.radius(3.0) == pytest.approx(2.0)

    def test_rejects_non_positive_wavelength(self):
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            Beam(
                beam_parameter=1j,
                z_offset=0.0,
                wavelength=0.0,
            )


class TestThinLens:
    def test_matrix_matches_thin_lens_transform(self):
        lens = ThinLens(focal_length=2.0)

        np.testing.assert_allclose(
            lens.matrix,
            np.array(
                [
                    [1.0, 0.0],
                    [-0.5, 1.0],
                ]
            ),
        )

    def test_rejects_zero_focal_length(self):
        with pytest.raises(ValueError, match="Focal length cannot be zero"):
            ThinLens(focal_length=0.0)

    def test_string_uses_name_when_provided(self):
        lens = ThinLens(focal_length=2.0, name="objective")

        assert str(lens) == "objective"


class TestOpticalSetup:
    def test_rejects_elements_out_of_position_order(self):
        beam = Beam.from_gauss(focus=0.0, waist=1.0, wavelength=1.0)
        lens = ThinLens(focal_length=2.0)

        with pytest.raises(ValueError, match="sorted by position"):
            OpticalSetup(
                initial_beam=beam,
                elements=[(2.0, lens), (1.0, lens)],
            )

    def test_propagates_beam_without_elements(self):
        beam = Beam.from_gauss(focus=0.0, waist=1.0, wavelength=1.0)
        setup = OpticalSetup(initial_beam=beam, elements=[])

        assert setup.beam_parameters == [beam.beam_parameter]
        assert setup.radius(0.0) == pytest.approx(beam.radius(0.0))
        assert setup.radius(1.0) == pytest.approx(beam.radius(1.0))

    def test_propagates_beam_through_lens(self):
        beam = Beam.from_gauss(focus=0.0, waist=1.0, wavelength=1.0)
        lens = ThinLens(focal_length=2.0)
        setup = OpticalSetup(initial_beam=beam, elements=[(1.0, lens)])

        assert len(setup.beams) == 2
        assert setup.beams[0] == beam
        assert setup.beams[1].z_offset == pytest.approx(1.0)
        assert setup.radius(1.0) == pytest.approx(setup.beams[1].radius(1.0))
