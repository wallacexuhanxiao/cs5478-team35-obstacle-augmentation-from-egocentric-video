"""Small backend-independent rigid-body reference-point calculations.

Only `origin_velocity` is used by the default Phase-1 execution path: PhysX
reports root COM linear velocity while the wrist PD position error is defined at
the root-link origin, so comparing those quantities without shifting the
reference point is a coordinate/kinematics bug.

`gravity_compensation` is retained as a HOLD diagnostic helper but is not used
by the default controller until an A/B test justifies changing the upstream
controller model.
"""


def cross(a, b, stack):
    return stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        )
    )


def origin_velocity(com_velocity, omega_world, origin_minus_com, stack):
    """Shift linear velocity from COM to another point on the rigid body."""
    return com_velocity + cross(omega_world, origin_minus_com, stack)


def gravity_compensation(masses, world_coms, stack):
    """HOLD helper: distributed gravity wrench about the root COM.

    This is intentionally not part of the default Phase-1 controller. Keeping
    the calculation here lets us test it separately if a later A/B diagnosis
    shows the upstream gravity feedforward is itself the root cause of a bug.
    """
    zeros = masses * 0.0
    forces = stack((zeros, zeros, masses * 9.81))
    arms = world_coms - world_coms[..., :1, :]
    return forces.sum(-2), cross(arms, forces, stack).sum(-2)
