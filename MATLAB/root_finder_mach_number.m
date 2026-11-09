function M = root_finder_mach_number(gamma, tol, M0, A, At, isSupersonic)
    if nargin < 6
        isSupersonic = M0 > 1;
    end

    n = (gamma+1)/(2*(gamma-1));
    AR_target = A/At;

    if abs(AR_target - 1) < 1e-4
        M = 1;
        return;
    end

    f = @(M) (1./M).*((2/(gamma+1))*(1+((gamma-1)/2)*M.^2)).^n - AR_target;

    if isSupersonic
        branch = 'supersonic';
        lower = 1 + 1e-6;
        upper = 60;
    else
        branch = 'subsonic';
        lower = 1e-6;
        upper = 1 - 1e-6;
    end

    flo = f(lower);
    fhi = f(upper);

    if sign(flo) == sign(fhi)
        if isSupersonic
            upper = 500;
            fhi = f(upper);
        end
        if sign(flo) == sign(fhi)
            error('nozzle:MachSolve', ...
                ['No %s Mach number satisfies A/At = %.6g (bracket [%g, %g] ' ...
                 'does not contain a root; A/At must be >= 1).'], ...
                branch, AR_target, lower, upper);
        end
    end

    opts = optimset('TolX', tol);
    try
        M = fzero(f, [lower, upper], opts);
    catch ME
        error('nozzle:MachSolve', 'fzero failed on the %s branch for A/At = %.6g: %s', ...
            branch, AR_target, ME.message);
    end
end
