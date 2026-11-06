function [M, exit_index, shock_index] = mach_number_ditribustion(nozzle_geometry, x, gamma, M0sub, M0super, tol,bow_shock)
Mach = zeros(size(x));
At = pi*(nozzle_geometry(7)/2)^2;
total_steps = numel(x);

throat_end = sum(nozzle_geometry(1:3));

fprintf('Mach solve:   0%%');
nextPct = 5;

for i = 1:total_steps
    if x(i) < nozzle_geometry(1)
        D = nozzle_geometry(6);
    elseif x(i) >= nozzle_geometry(1) && x(i) < sum(nozzle_geometry(1:2))
        D = ((nozzle_geometry(7)-nozzle_geometry(6))/nozzle_geometry(2))*(x(i)-nozzle_geometry(1)) + nozzle_geometry(6);
    elseif x(i) >= sum(nozzle_geometry(1:2)) && x(i) < sum(nozzle_geometry(1:3))
        D = nozzle_geometry(7);
    elseif x(i) >= sum(nozzle_geometry(1:3)) && x(i) < sum(nozzle_geometry(1:4))
        D = ((nozzle_geometry(8)-nozzle_geometry(7))/nozzle_geometry(4))*(x(i)-sum(nozzle_geometry(1:3))) + nozzle_geometry(7);
    elseif x(i) >= sum(nozzle_geometry(1:4))
        D = nozzle_geometry(8);
    end

    A = pi*(D/2)^2;
    isSupersonic = x(i) >= throat_end;

    if x(i) == 0
        M0 = M0sub;
    elseif x(i) == throat_end
        M0 = M0super;
    elseif x(i) >= sum(nozzle_geometry(1:2)) && x(i) < sum(nozzle_geometry(1:3))
        M0 = 1.001;
    else
        M0 = Mach(i-1);
    end

    Mach(i) = root_finder_mach_number(gamma, tol, M0, A, At, isSupersonic);

    pct = 100*i/total_steps;
    if pct >= nextPct
        fprintf('\b\b\b\b%3d%%', round(pct));
        nextPct = nextPct + 5;
    end
end
fprintf('\n');

[~, exit_index] = min(abs(x - sum(nozzle_geometry(1:4))));
shock_index = [];

if bow_shock
    mach_exit = Mach(exit_index);
    L_shock = 0.143*nozzle_geometry(8)*exp(3.24/mach_exit^2);
    M_shock = normal_shock(mach_exit, gamma);
    [ ~ ,shock_index] = min(abs(x - (x(end) - L_shock)));
    M_after_shock = linspace(M_shock,0,numel(x) + 1 - shock_index);
    Mach(shock_index:end) = M_after_shock;
end

M = Mach;
end