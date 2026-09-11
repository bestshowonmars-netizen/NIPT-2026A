"""Conservative radial cell-centred FVM; backward Euler and Picard, no extrapolation."""
import numpy as np
from numba import njit
from common.material_properties import properties
from common.boundary_conditions import surface_conductance, surface_value
from common.interpolation import environment_at

@njit(cache=True)
def geometry(n, radius):
    x = np.linspace(0.0, 1.0, n + 1)
    faces = radius * (10.0*x - x**10) / 9.0  # gentle interior spacing, refined surface layer
    volumes = (faces[1:]**2 - faces[:-1]**2) / 2.0  # omit common 2*pi*L
    centers = (2.0 / 3.0) * (faces[1:]**3 - faces[:-1]**3) / (faces[1:]**2 - faces[:-1]**2)
    return faces, centers, volumes

@njit(cache=True)
def fill_properties(T, C, question, a, k, D):
    for i in range(len(T)):
        a[i], k[i], D[i] = properties(T[i], C[i], question)

@njit(cache=True)
def links(coef, exchange, faces, centers, out):
    n = len(coef)
    out[0] = 0.0
    for j in range(1, n):
        out[j] = faces[j] / ((faces[j] - centers[j-1]) / coef[j-1] + (centers[j] - faces[j]) / coef[j])
    out[n] = surface_conductance(coef[-1], exchange, faces[-1]-centers[-1], faces[-1])

@njit(cache=True)
def linear_step(old, capacity, volume, conductance, external, dt, out, work):
    # Solve for the increment instead of absolute Kelvin values to avoid cancellation.
    n = len(old)
    upper, modified_rhs = work[0], work[1]
    for i in range(n):
        scale = dt / (capacity[i] * volume[i])
        left, right = conductance[i] * scale, conductance[i+1] * scale
        diagonal = 1.0 + left + right
        rhs = 0.0
        if i > 0:
            rhs += left * (old[i-1] - old[i])
        if i < n - 1:
            rhs += right * (old[i+1] - old[i])
        else:
            rhs += right * (external - old[i])
        if i > 0:
            diagonal -= left * upper[i-1]
            rhs += left * modified_rhs[i-1]
        upper[i] = right / diagonal
        modified_rhs[i] = rhs / diagonal
    increment = modified_rhs[-1]
    out[-1] = old[-1] + increment
    for i in range(n-2, -1, -1):
        increment = modified_rhs[i] + upper[i] * increment
        out[i] = old[i] + increment

@njit(cache=True)
def residual(old, new, capacity, volume, conductance, external, dt, atol, rtol):
    largest = 0.0
    for i in range(len(old)):
        mass = capacity[i] * volume[i] / dt
        flux = 0.0
        if i > 0:
            flux += conductance[i] * (new[i-1] - new[i])
        if i < len(old) - 1:
            flux += conductance[i+1] * (new[i+1] - new[i])
        else:
            flux += conductance[i+1] * (external - new[i])
        denom = (mass + conductance[i] + conductance[i+1]) * (atol + rtol*abs(new[i]))
        largest = max(largest, abs(mass*(new[i]-old[i])-flux)/denom)
    return largest

@njit(cache=True)
def reconstruct(values, coef, exchange, external, faces, centers, sample_r):
    # Even quadratic exactly matches the first two cylindrical volume averages.
    q0 = 0.5*(faces[0]**2 + faces[1]**2)
    q1 = 0.5*(faces[1]**2 + faces[2]**2)
    beta = (values[1]-values[0])/(q1-q0)
    center_value = values[0] - beta*q0
    surface = surface_value(values[-1], external, coef[-1], exchange, faces[-1]-centers[-1])
    xs = np.empty(len(values)+2)
    ys = np.empty(len(values)+2)
    xs[0], xs[-1] = 0.0, faces[-1]
    ys[0], ys[-1] = center_value, surface
    xs[1:-1], ys[1:-1] = centers, values
    result = np.interp(sample_r, xs, ys)
    for j in range(len(sample_r)):
        if sample_r[j] < centers[0]:
            result[j] = center_value + beta*sample_r[j]**2
    return result

@njit(cache=True)
def integrate_block(T, C, t_start, end_second, dt_max, question, env,
                    faces, centers, volume, sample_r, h, hm, atol, rtol, max_iterations, time_growth=0.0):
    n = len(T)
    count = end_second-int(t_start)
    ts = np.empty((count, len(sample_r)))
    cs = np.empty_like(ts)
    # columns: normalized mean C, normalized loss in this second, steps, iterations,
    # rejections, min dt, max dt, max scaled residual, damped iterations
    log = np.zeros((count, 9))
    a, k, D = np.empty(n), np.empty(n), np.empty(n)
    ones = np.ones(n)
    gt, gc = np.empty(n+1), np.empty(n+1)
    work = np.empty((2,n))
    guessT, guessC, nextT, nextC = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    total_volume = np.sum(volume)
    t = t_start
    for row in range(count):
        target = t_start+row+1
        loss, steps, iterations, rejected, minimum, maximum, worst, damped = 0.0,0,0,0,1e100,0.0,0.0,0
        dt_limit = min(1.0,4.0*dt_max) if time_growth > 0.0 else dt_max
        while t < target-1e-12:
            requested_dt = dt_max * (1.0+t/time_growth)**2 if time_growth > 0.0 else dt_max
            dt = min(dt_limit, requested_dt, target-t)
            # Attachment knots lie at whole seconds; integer output fences also respect them.
            extT, extC = environment_at(t+dt,env)
            guessT[:], guessC[:] = T, C
            converged = False
            previous_change = 1e100
            omega = 1.0
            for it in range(max_iterations):
                fill_properties(guessT,guessC,question,a,k,D)
                if question == 2 or it == 0:
                    links(k,h,faces,centers,gt)
                    linear_step(T,a,volume,gt,extT,dt,nextT,work)
                fill_properties(nextT,guessC,question,a,k,D)
                links(D,hm,faces,centers,gc)
                linear_step(C,ones,volume,gc,extC,dt,nextC,work)
                if not np.all(np.isfinite(nextT)) or not np.all(np.isfinite(nextC)):
                    break
                if np.min(nextC) <= 0.0 or np.min(nextT) <= 0.0:
                    omega *= 0.5
                change = 0.0
                for j in range(n):
                    change=max(change,abs(nextT[j]-guessT[j])/(atol+rtol*abs(nextT[j])))
                    change=max(change,abs(nextC[j]-guessC[j])/(atol+rtol*abs(nextC[j])))
                if change > previous_change*1.2:
                    omega = max(0.125,omega*0.5)
                if omega < 1.0:
                    damped += 1
                    nextT[:] = guessT + omega*(nextT-guessT)
                    nextC[:] = guessC + omega*(nextC-guessC)
                if np.min(nextC) <= 0.0 or np.min(nextT) <= 0.0:
                    break
                if change <= 1.0:
                    fill_properties(nextT,nextC,question,a,k,D)
                    links(k,h,faces,centers,gt)
                    links(D,hm,faces,centers,gc)
                    rt = residual(T,nextT,a,volume,gt,extT,dt,atol,rtol)
                    rc = residual(C,nextC,ones,volume,gc,extC,dt,atol,rtol)
                    if max(rt,rc) <= 1.0:
                        converged=True
                        worst=max(worst,rt,rc)
                        break
                guessT[:],guessC[:] = nextT,nextC
                previous_change=change
            iterations += it+1
            if not converged:
                rejected += 1
                dt_limit=dt*0.5
                if dt_limit < 1e-8:
                    raise RuntimeError("Picard failed below minimum time step; no state clipping applied")
                continue
            loss += dt*gc[-1]*(nextC[-1]-extC)/total_volume
            T[:],C[:] = nextT,nextC
            t += dt
            minimum,maximum=min(minimum,dt),max(maximum,dt)
            steps += 1
        t=target
        fill_properties(T,C,question,a,k,D)
        extT,extC=environment_at(t,env)
        ts[row]=reconstruct(T,k,h,extT,faces,centers,sample_r)
        cs[row]=reconstruct(C,D,hm,extC,faces,centers,sample_r)
        log[row,0]=np.sum(volume*C)/total_volume
        log[row,1]=loss
        log[row,2]=steps
        log[row,3]=iterations
        log[row,4]=rejected
        log[row,5]=minimum
        log[row,6]=maximum
        log[row,7]=worst
        log[row,8]=damped
    return T,C,ts,cs,log
