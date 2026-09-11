"""State-preserving continuation of the existing appendix-3 finite-volume model.

Absolute time is retained.  The original Q1/Q2 kernel is left unchanged.
Each accepted step uses the same backward Euler, Picard update, Robin
conductance, residual test, and implicit water flux as finite_volume.py.
"""
import numpy as np
from numba import njit
from common.finite_volume import fill_properties, links, linear_step, residual, reconstruct
from common.interpolation import environment_at


@njit(cache=True)
def advance_interval(T0, C0, t_start, t_end, dt_max, env, faces, centers, volume,
                     h, hm, atol, rtol, max_iterations):
    """Return new full-grid states and 9 diagnostics; never mutate input state.

    Diagnostic order agrees with integrate_block:
    mean_C, interval_loss, steps, iterations, rejections, min_dt, max_dt,
    worst_scaled_residual, damped_iterations.
    """
    if t_end < t_start or dt_max <= 0.0 or max_iterations < 1:
        raise ValueError("Invalid continuation interval or step configuration")
    T, C = T0.copy(), C0.copy()
    n = len(T)
    a, k, D = np.empty(n), np.empty(n), np.empty(n)
    ones = np.ones(n)
    gt, gc = np.empty(n + 1), np.empty(n + 1)
    work = np.empty((2, n))
    guessT, guessC = np.empty(n), np.empty(n)
    nextT, nextC = np.empty(n), np.empty(n)
    total_volume = np.sum(volume)
    t, dt_limit = t_start, dt_max
    loss, steps, iterations, rejected = 0.0, 0, 0, 0
    minimum, maximum, worst, damped = 1e100, 0.0, 0.0, 0
    while t < t_end - 1e-12:
        # Respect every actual environment knot, including the 4-hour endpoint.
        # This also supports a future input file with non-integer knot times.
        knot_index = np.searchsorted(env[:, 0], t + 1e-10, side="right")
        next_knot = env[knot_index, 0] if knot_index < len(env) else t_end
        dt = min(dt_limit, t_end - t, next_knot - t)
        extT, extC = environment_at(t + dt, env)
        guessT[:], guessC[:] = T, C
        converged, previous_change, omega = False, 1e100, 1.0
        for it in range(max_iterations):
            fill_properties(guessT, guessC, 2, a, k, D)
            links(k, h, faces, centers, gt)
            linear_step(T, a, volume, gt, extT, dt, nextT, work)
            fill_properties(nextT, guessC, 2, a, k, D)
            links(D, hm, faces, centers, gc)
            linear_step(C, ones, volume, gc, extC, dt, nextC, work)
            if not np.all(np.isfinite(nextT)) or not np.all(np.isfinite(nextC)):
                break
            if np.min(nextC) <= 0.0 or np.min(nextT) <= 0.0:
                omega *= 0.5
            change = 0.0
            for j in range(n):
                change = max(change, abs(nextT[j]-guessT[j])/(atol+rtol*abs(nextT[j])))
                change = max(change, abs(nextC[j]-guessC[j])/(atol+rtol*abs(nextC[j])))
            if change > previous_change * 1.2:
                omega = max(0.125, omega * 0.5)
            if omega < 1.0:
                damped += 1
                nextT[:] = guessT + omega * (nextT - guessT)
                nextC[:] = guessC + omega * (nextC - guessC)
            if np.min(nextC) <= 0.0 or np.min(nextT) <= 0.0:
                break
            if change <= 1.0:
                fill_properties(nextT, nextC, 2, a, k, D)
                links(k, h, faces, centers, gt)
                links(D, hm, faces, centers, gc)
                rt = residual(T, nextT, a, volume, gt, extT, dt, atol, rtol)
                rc = residual(C, nextC, ones, volume, gc, extC, dt, atol, rtol)
                if max(rt, rc) <= 1.0:
                    converged = True
                    worst = max(worst, rt, rc)
                    break
            guessT[:], guessC[:] = nextT, nextC
            previous_change = change
        iterations += it + 1
        if not converged:
            rejected += 1
            dt_limit = dt * 0.5
            if dt_limit < 1e-8:
                raise RuntimeError("Q3 Picard failed below minimum step; no clipping")
            continue
        loss += dt * gc[-1] * (nextC[-1] - extC) / total_volume
        T[:], C[:] = nextT, nextC
        t += dt
        minimum, maximum = min(minimum, dt), max(maximum, dt)
        steps += 1
    diag = np.array([np.sum(volume*C)/total_volume, loss, float(steps),
                     float(iterations), float(rejected), minimum if steps else 0.0,
                     maximum, worst, float(damped)])
    return T, C, diag


@njit(cache=True)
def sample_fields(T, C, t, env, faces, centers, sample_r, h, hm):
    """Reconstruct output positions; full-grid states remain the solver state."""
    a, k, D = np.empty(len(T)), np.empty(len(T)), np.empty(len(T))
    fill_properties(T, C, 2, a, k, D)
    extT, extC = environment_at(t, env)
    return (reconstruct(T, k, h, extT, faces, centers, sample_r),
            reconstruct(C, D, hm, extC, faces, centers, sample_r))


@njit(cache=True)
def global_maximum(T, C, t, env, faces, centers, h, hm):
    """Maximum over all cells plus the symmetric center and Robin surface."""
    bounds = np.array([0.0, faces[-1]])
    _, boundary_C = sample_fields(T, C, t, env, faces, centers, bounds, h, hm)
    maximum, location = boundary_C[0], 0.0
    for j in range(len(C)):
        if C[j] > maximum:
            maximum, location = C[j], centers[j]
    if boundary_C[1] > maximum:
        maximum, location = boundary_C[1], faces[-1]
    return maximum, location, boundary_C[0], boundary_C[1]

