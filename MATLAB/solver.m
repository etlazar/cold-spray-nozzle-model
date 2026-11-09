clc; clear; close all;
tic;

% Nozzle Geometry
Li = 5; % mm
Lc = 17.02; % mm
Lt = 2.29; % mm
Ld = 150.62; % mm
Ls = 25.4; % mm
Di = 9.52; % mm
Dt = 1.98; % mm
De = 3.81; % mm

% Gas Parameters
pressure0 = 900; % psi
pressure_is_gauge = false; % true: pressure0 is psig, 1 atm is added
temp0 = 650; % C
gamma = 1.4;
cp = 1.04; % kJ/kg*K
R = 296.8;
Cs = 107;
ref_temp = 273;
ref_viscosity = 1.66e-5;


% Powder Parameters
p_density = 7310; % kg/m^3
dp10 = 2; % μm
dp50 = 10; % μm
dp90 = 30; % μm
psi = 0.5;
vp0 = 0.1; % m/s ([] = start at the inlet gas velocity)
p_temp0 = 25; % C
% ult_strength = 16; % MPa
% melting_temp = 250; % C
cp_par = 227;
% k_p = 67;
% L_f = 59000;
% E_p = 50*1e9;
% nu_p = 0.36;
% surf_ten = 0.560;
% mu_liq = 1.85e-3;


% Substrate Properties
% nu_sub = 0.375;
% E_sub = 3.5*1e9;
% K_IC = 1.25;
% c = 15e-16;
% theta = 130;
% T_sub = 273;


% Solver Parameters
h = 0.001; % mm
int_M0 = 0.03;
post_throat_M0 = 1.05;
tol = 1e-5;
bow_shock = true;
standoff_length_scale = 'diameter'; % exit dimension in Billig's correlation: 'diameter' or 'radius'

% Consolidations, Calculations, and Conversions
pressure0 = pressure0 * 6894.757;
if pressure_is_gauge
    pressure0 = pressure0 + 101325;
end
temp0 = temp0 + 273.15;
dp10 = dp10*10^(-6);
dp50 = dp50*10^(-6);
dp90 = dp90*10^(-6);
p_temp0 = p_temp0 + 273.15;
% melting_temp = melting_temp + 273.15;
% ult_strength = ult_strength*1e6;
gas_density0 = pressure0/(R*temp0);
cp = cp*1e3; % kJ/kg*K -> J/kg*K
nozzle_geometry = [Li Lc Lt Ld Ls Di Dt De];
Ltotal = sum(nozzle_geometry(1:5));
x = 0:h:Ltotal;

wb = waitbar(0, 'Starting...', 'Name', 'Nozzle Solver');
nStages = 9;

% Find the mach number distribution
waitbar(0/nStages, wb, 'Solving Mach distribution...');
try
    [M, exit_index, shock_index] = mach_number_ditribustion(nozzle_geometry,x,gamma,int_M0,post_throat_M0,tol,bow_shock,standoff_length_scale);
catch ME
    close(wb);
    rethrow(ME);
end

% Find the properties of the gas
waitbar(1/nStages, wb, 'Computing gas properties...');
temp = find_property('temp',temp0,M,gamma);
pressure = find_property('pressure',pressure0,M,gamma);
gas_density = find_property('density',gas_density0,M,gamma);
v = find_property('velocity', temp0 ,M,[gamma R]);
viscosity = find_property('viscosity', ref_viscosity, temp, [ref_temp Cs]);

% The normal shock is not isentropic: T0 is conserved but p0 drops, so
% everything behind the shock decelerates isentropically from p02
if ~isempty(shock_index)
    [~, ~, p0_ratio] = normal_shock(M(exit_index), gamma);
    post = shock_index:numel(x);
    pressure(post) = find_property('pressure',pressure0*p0_ratio,M(post),gamma);
    gas_density(post) = find_property('density',gas_density0*p0_ratio,M(post),gamma);
end

% dvp/dx ~ 1/vp, so starting near rest makes the first steps stiff;
% vp0 = [] starts the particles at the local gas velocity instead
if isempty(vp0)
    vp0 = v(1);
end

% Find the velocity of the particles
waitbar(2/nStages, wb, 'Particle velocity: D10...');
p_props10 = RK4_vp(gas_density,v,dp10,h,x,viscosity,p_density,vp0);
Rep10 = p_props10(1,:);
Cd10 = p_props10(2,:);
vp10 = p_props10(3,:);

waitbar(3/nStages, wb, 'Particle velocity: D50...');
p_props50 = RK4_vp(gas_density,v,dp50,h,x,viscosity,p_density,vp0);
Rep50 = p_props50(1,:);
Cd50 = p_props50(2,:);
vp50 = p_props50(3,:);

waitbar(4/nStages, wb, 'Particle velocity: D90...');
p_props90 = RK4_vp(gas_density,v,dp90,h,x,viscosity,p_density,vp0);
Rep90 = p_props90(1,:);
Cd90 = p_props90(2,:);
vp90 = p_props90(3,:);


% Calculate the Properties need to solve the heat transfer ode
waitbar(5/nStages, wb, 'Heat transfer properties...');
kg = find_property('kg', 0 ,temp);
Pr = find_property('Pr', 0, [viscosity; kg],cp);
Nu10 = find_property('Nu', 0, [Rep10; Pr]);
Nu50 = find_property('Nu', 0, [Rep50; Pr]);
Nu90 = find_property('Nu', 0, [Rep90; Pr]);
hp10 = find_property('hp', 0, [Nu10; kg], dp10);
hp50 = find_property('hp', 0, [Nu50; kg], dp50);
hp90 = find_property('hp', 0, [Nu90; kg], dp90);

% Finds temp of the particles
waitbar(6/nStages, wb, 'Particle temperature: D10...');
Tp10 = RK4_Tp(cp_par,dp10,hp10,temp,p_temp0,x,h,p_density,vp10);
waitbar(7/nStages, wb, 'Particle temperature: D50...');
Tp50 = RK4_Tp(cp_par,dp50,hp50,temp,p_temp0,x,h,p_density,vp50);
waitbar(8/nStages, wb, 'Particle temperature: D90...');
Tp90 = RK4_Tp(cp_par,dp90,hp90,temp,p_temp0,x,h,p_density,vp90);

waitbar(1, wb, 'Done');
pause(0.3);
close(wb);
fprintf('Total time: %.2f s\n', toc);

% Explicit RK4 goes unstable for small particles when the step is too large.
% Physically vp stays in [0, ~v_max] and Tp between the coldest of (its start,
% the gas) and the stagnation temperature T0 (1 K tolerance).
Tp_tol = 1;
Tp_min = min(min(temp), p_temp0) - Tp_tol;
Tp_max = temp0 + Tp_tol;
names = {'D10', 'D50', 'D90'};
vps = {vp10, vp50, vp90};
Tps = {Tp10, Tp50, Tp90};
unstable = {};
for k = 1:3
    if ~all(isfinite(vps{k})) || ~all(isfinite(Tps{k})) ...
            || min(vps{k}) < 0 || max(vps{k}) > 1.5*max(v) ...
            || min(Tps{k}) < Tp_min || max(Tps{k}) > Tp_max
        unstable{end+1} = names{k}; %#ok<SAGROW>
    end
end
if ~isempty(unstable)
    warning('nozzle:Unstable', ['The particle solution for %s is unstable at a step size of %g mm ' ...
        '(non-physical velocity or temperature). Reduce the step size (e.g. <= 0.005 mm) ' ...
        'or set vp0 = [] and rerun.'], strjoin(unstable, ', '), h);
end

% Plot Nozzle
plot_nozzle(nozzle_geometry,h);

% Plot and Compare gas and particle properties

figure;
set(gcf, "Theme", "light")
plot(x,temp);
hold on;
plot (x,Tp10);
plot(x,Tp50)
plot(x,Tp90)
hold off;
title("Particle and Gas Tempature Across the Nozzle")
ylabel('Tempature [K]');
xlabel('Axial Postion [mm]');
legend('Gas', "Particle D10", "Particle D50","Particle D90");
xlim([x(1) x(end)]);

figure;
set(gcf, "Theme", "light");
plot(x, v);
hold on;
plot(x, vp10);
plot(x,vp50)
plot(x,vp90)
hold off;
title("Gas and Particle Velocity Across the Nozzle");
ylabel("Velocity [m/s]");
xlabel("Axial Position [mm]");
legend("Gas", "Particle D10", "Particle D50","Particle D90");
xlim([x(1) x(end)]);

figure;
plot(x,pressure);
title("Gas Pressure");
ylabel('Pressure [Pa]');
xlabel('Axial Postion [mm]');
set(gcf, "Theme", "light");
xlim([x(1) x(end)]);

figure;
plot(x,M);
title("Mach Number");
ylabel("Mach Number");
xlabel("Axial Position [mm]");
grid on;
set(gcf, "Theme", "light");
xlim([x(1) x(end)]);