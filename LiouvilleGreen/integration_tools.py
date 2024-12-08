from scipy.optimize import root_scalar


def find_phase_minimum(
    sol, start_z: float, stop_z: float, value_index: int, deriv_index: int
):
    start_deriv = sol(start_z)[deriv_index]
    last_deriv = start_deriv

    stepsize = -(1e-3) * start_z
    found_zero = False

    last_z = start_z
    current_z = start_z + stepsize
    current_deriv = None
    while current_z > stop_z:
        current_deriv = sol(current_z)[deriv_index]

        # has there been a sign change since the last time we sampled Gprime?
        if current_deriv * last_deriv < 0 and last_deriv < 0:
            found_zero = True
            break

        last_z = current_z

        stepsize = -(1e-3) * current_z
        current_z += stepsize
        last_deriv = current_deriv

    if not found_zero:
        raise RuntimeError(
            f"Did not find zero of derivative in the search window (start_z={start_z:.5g}, stop_z={stop_z:.5g}), current_z={current_z:.5g}, start Gprime={start_deriv:.5g}, last derivative={last_deriv:.5g}, current derivative={current_deriv:.5g}"
        )

    root = root_scalar(
        lambda z: sol(z)[deriv_index],
        bracket=(last_z, current_z),
        xtol=1e-6,
        rtol=1e-4,
    )

    if not root.converged:
        raise RuntimeError(
            f'root_scalar() did not converge to a solution: z_bracket=({last_z:.5g}, {current_z:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    root_z = root.root
    sol_root = sol(root_z)

    return {
        "z": root_z,
        "value": sol_root[value_index],
        "derivative": sol_root[deriv_index],
    }
