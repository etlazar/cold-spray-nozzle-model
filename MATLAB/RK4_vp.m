function [out] = RK4_vp(gas_density, v, dp, h, x, viscosity, p_density, vp0)
Rep = zeros(size(x));
Cd  = zeros(size(x));
vp  = zeros(size(x));
vp(1) = vp0;
h = h*1e-3;
mp = p_density * (4/3) * pi * (dp/2)^3;
Ap = pi * (dp/2)^2;
dvp = @(Cd,rho,v,vp) (Cd*rho*Ap)/(2*mp*vp) * abs(v - vp) * (v - vp);
Cd_fun = @(Rep) (24./Rep) .* (1 + 0.15.*Rep.^0.687) + 0.42 ./ (1 + 4.25e4.*Rep.^(-1.16));
N = numel(x)-1;
fprintf('Particle velocity:   0%%');
nextPct = 5;
for i = 1:N
    Rep(i) = gas_density(i) * abs(v(i) - vp(i)) * dp / viscosity(i);
    Cd(i)  = Cd_fun(Rep(i));
    gas_density_half = (gas_density(i) + gas_density(i+1)) / 2;
    v_half           = (v(i) + v(i+1)) / 2;
    viscosity_half   = (viscosity(i) + viscosity(i+1)) / 2;
    gas_density_full = gas_density(i+1);
    v_full           = v(i+1);
    viscosity_full   = viscosity(i+1);
    k1 = dvp(Cd(i), gas_density(i), v(i), vp(i));
    vp_k2 = vp(i) + 0.5*h*k1;
    Rep_half = gas_density_half * abs(v_half - vp_k2) * dp / viscosity_half;
    Cd_half  = Cd_fun(Rep_half);
    k2 = dvp(Cd_half, gas_density_half, v_half, vp_k2);
    vp_k3 = vp(i) + 0.5*h*k2;
    Rep_half3 = gas_density_half * abs(v_half - vp_k3) * dp / viscosity_half;
    Cd_half3  = Cd_fun(Rep_half3);
    k3 = dvp(Cd_half3, gas_density_half, v_half, vp_k3);
    vp_k4 = vp(i) + h*k3;
    Rep_full = gas_density_full * abs(v_full - vp_k4) * dp / viscosity_full;
    Cd_full  = Cd_fun(Rep_full);
    k4 = dvp(Cd_full, gas_density_full, v_full, vp_k4);
    vp(i+1) = vp(i) + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    pct = 100*i/N;
    if pct >= nextPct
        fprintf('\b\b\b\b%3d%%', round(pct));
        nextPct = nextPct + 5;
    end
end
fprintf('\n');
Rep(end) = gas_density(end) * abs(v(end) - vp(end)) * dp / viscosity(end);
Cd(end)  = Cd_fun(Rep(end));
out = [Rep; Cd; vp];
end