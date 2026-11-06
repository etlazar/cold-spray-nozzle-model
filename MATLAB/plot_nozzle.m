function plot_nozzle(nozzle_geometry, h, ax)
% plot_nozzle(nozzle_geometry, h, ax)
%
% ax (optional): handle to axes to plot into.
%   - If omitted, a new figure/axes is created (original behavior).
%   - If provided, plots into that axes (e.g. a subplot, tiledlayout,
%     uiaxes in an app, or an existing figure's axes).

if nargin < 3 || isempty(ax)
    f = figure;
    ax = axes(f);
end

title(ax, "Nozzle");
hold(ax, 'on');

% Inlet
region = 0:h:nozzle_geometry(1);
r = nozzle_geometry(6)/2 * ones(size(region));
plot(ax, region, r, 'r');
plot(ax, region, -r, 'r');

% Converging
region = nozzle_geometry(1):h:sum(nozzle_geometry(1:2));
m = (nozzle_geometry(7)/2 - nozzle_geometry(6)/2) / nozzle_geometry(2);
r = nozzle_geometry(6)/2 + m * (region - nozzle_geometry(1));
plot(ax, region, r, 'r');
plot(ax, region, -r, 'r');

% Throat
region = sum(nozzle_geometry(1:2)):h:sum(nozzle_geometry(1:3));
r = nozzle_geometry(7)/2 * ones(size(region));
plot(ax, region, r, 'r');
plot(ax, region, -r, 'r');

% Diverging
region = sum(nozzle_geometry(1:3)):h:sum(nozzle_geometry(1:4));
m = (nozzle_geometry(8)/2 - nozzle_geometry(7)/2) / nozzle_geometry(4);
r = nozzle_geometry(7)/2 + m * (region - sum(nozzle_geometry(1:3)));
plot(ax, region, r, 'r');
plot(ax, region, -r, 'r');

% Standoff
region = sum(nozzle_geometry(1:4)):h:sum(nozzle_geometry(1:5));
r = nozzle_geometry(8)/2 * ones(size(region));
plot(ax, region, r, 'k:');
plot(ax, region, -r, 'k:');

axis(ax, 'equal');
hold(ax, 'off');

ax.YAxis.Visible = 'off';
set(ancestor(ax, 'figure'), "Theme", "light");
axis(ax, 'off');
end