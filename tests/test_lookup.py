import unittest

from fishery.config import FisheryConfig
from fishery.lookup import linear_interpolate


class LookupTests(unittest.TestCase):
    def test_death_fraction_knots_are_exact(self) -> None:
        config = FisheryConfig()
        for x_value, expected in config.death_fraction_points:
            self.assertAlmostEqual(
                linear_interpolate(x_value, config.death_fraction_points),
                expected,
            )

    def test_catch_density_knots_are_exact(self) -> None:
        config = FisheryConfig()
        for x_value, expected in config.catch_in_density_points:
            self.assertAlmostEqual(
                linear_interpolate(x_value, config.catch_in_density_points),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
