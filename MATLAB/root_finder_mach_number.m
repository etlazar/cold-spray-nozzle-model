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
        lower = 1 + 1e-6;
        upper = 60;
    else
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
            M = M0;
            return;
        end
    end

    opts = optimset('TolX', tol);
    try
        M = fzero(f, [lower, upper], opts);
    catch ME
        M = M0;
    end
end