"""Appendix-4 coupled implicit FVM in a fixed material reference domain."""
import numpy as np
from numba import njit
from common.finite_volume import geometry, links, linear_step, residual, reconstruct
from common.interpolation import environment_at
from q4.shrinking_radius import radius_at


@njit(cache=True)
def fill_properties4(T, C, scale, a, k, D):
    """Appendix 4; scale transports once on the fixed reference domain."""
    for j in range(len(T)):
        c = C[j]
        fraction = c/(1.0+c)
        a[j] = (760.0+90.0*c)*(1850.0+2150.0*fraction)
        k[j] = (0.12+0.20*fraction)/(scale*scale)
        D[j] = 4.2e-4*np.exp(-0.30/c)*np.exp(-3850.0/T[j])/(scale*scale)


@njit(cache=True)
def advance_interval(T0, C0, t_start, t_end, dt_max, env, faces, centers, volume,
                     h, hm, atol, rtol, max_iterations, radius_data, radius_slopes,
                     shrinking=True, dt_scale=1.0/768.0, time_growth=600.0):
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
        radius_index = np.searchsorted(radius_data[:, 0], t+1e-10, side="right")
        radius_knot = radius_data[radius_index, 0] if radius_index < len(radius_data) else t_end
        requested = dt_max if time_growth <= 0 else dt_scale*(1.0+t/time_growth)**2
        dt = min(dt_limit, requested, t_end - t, next_knot - t, radius_knot-t)
        scale = radius_at(t+dt, radius_data, radius_slopes, shrinking, faces[-1])/faces[-1]
        href, hmref = h/scale, hm/scale
        extT, extC = environment_at(t + dt, env)
        guessT[:], guessC[:] = T, C
        converged, previous_change, omega = False, 1e100, 1.0
        for it in range(max_iterations):
            fill_properties4(guessT, guessC, scale, a, k, D)
            links(k, href, faces, centers, gt)
            linear_step(T, a, volume, gt, extT, dt, nextT, work)
            fill_properties4(nextT, guessC, scale, a, k, D)
            links(D, hmref, faces, centers, gc)
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
                fill_properties4(nextT, nextC, scale, a, k, D)
                links(k, href, faces, centers, gt)
                links(D, hmref, faces, centers, gc)
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
                raise RuntimeError("Q4 Picard failed below minimum step; no clipping")
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
def sample_fields(T, C, t, env, faces, centers, sample_r, h, hm,
                  radius_data, radius_slopes, shrinking=True, reference=False):
    """Mask physical exterior BEFORE reconstruction; reference points are metres."""
    R = radius_at(t, radius_data, radius_slopes, shrinking, faces[-1])
    scale = R/faces[-1]
    mask = (sample_r >= 0.0) & (sample_r <= (faces[-1] if reference else R))
    indices = np.nonzero(mask)[0]
    query = sample_r[indices].copy()
    if not reference:
        query /= scale
    outT, outC = np.full(len(sample_r), np.nan), np.full(len(sample_r), np.nan)
    if t == 0.0 and np.max(T) == np.min(T) and np.max(C) == np.min(C):
        outT[indices], outC[indices] = T[0], C[0]
        return outT, outC, mask
    a, k, D = np.empty(len(T)), np.empty(len(T)), np.empty(len(T))
    fill_properties4(T, C, scale, a, k, D)
    extT, extC = environment_at(t, env)
    outT[indices] = reconstruct(T, k, h/scale, extT, faces, centers, query)
    outC[indices] = reconstruct(C, D, hm/scale, extC, faces, centers, query)
    return outT, outC, mask


@njit(cache=True)
def surface_fields(T, C, t, env, faces, centers, h, hm,
                   radius_data, radius_slopes, shrinking=True):
    ts, cs, _ = sample_fields(T, C, t, env, faces, centers,
                              np.array([faces[-1]]), h, hm,
                              radius_data, radius_slopes, shrinking, True)
    return ts[0], cs[0]


@njit(cache=True)
def global_maximum(T, C, t, env, faces, centers, h, hm,
                   radius_data, radius_slopes, shrinking=True):
    _, bounds, _ = sample_fields(T, C, t, env, faces, centers,
                                 np.array([0.0, faces[-1]]), h, hm,
                                 radius_data, radius_slopes, shrinking, True)
    maximum, location = bounds[0], 0.0
    for j in range(len(C)):
        if C[j] > maximum:
            maximum, location = C[j], centers[j]
    if bounds[1] > maximum:
        maximum, location = bounds[1], faces[-1]
    scale = radius_at(t, radius_data, radius_slopes, shrinking, faces[-1])/faces[-1]
    return maximum, scale*location, bounds[0], bounds[1]
