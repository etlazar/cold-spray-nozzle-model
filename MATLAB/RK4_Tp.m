function Tp = RK4_Tp(cp,dp,hp,temp,p_temp0,x,h,p_density,vp)
Tp = zeros(size(x));
Tp(1) = p_temp0;
mp = p_density * (4/3) * pi * (dp/2)^3;
Ap = pi * dp^2;
dTp = @(hp,temp,Tp,vp) (hp/(cp*vp))*(Ap/mp)*(temp-Tp);
h = h*1e-3;
N = numel(x) - 1;
fprintf('Particle temperature:   0%%');
nextPct = 5;
for i = 1:N
    temp_half_step = (temp(i)+temp(i+1))/2;
    hp_half_step = (hp(i)+hp(i+1))/2;
    vp_half_step = (vp(i)+vp(i+1))/2;
    k1 = dTp(hp(i),temp(i),Tp(i),vp(i));
    k2 = dTp(hp_half_step,temp_half_step,Tp(i)+(0.5*h*k1),vp_half_step);
    k3 = dTp(hp_half_step,temp_half_step,Tp(i)+(0.5*h*k2),vp_half_step);
    k4 = dTp(hp(i+1),temp(i+1),Tp(i)+(h*k3),vp(i+1));
    Tp(i+1) = Tp(i) + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    pct = 100*i/N;
    if pct >= nextPct
        fprintf('\b\b\b\b%3d%%', round(pct));
        nextPct = nextPct + 5;
    end
end
fprintf('\n');
end