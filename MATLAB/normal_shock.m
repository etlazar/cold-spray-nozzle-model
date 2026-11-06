function [M2, p_ratio, p0_ratio] = normal_shock(M1, gamma)
% Normal-shock relations: downstream Mach, p2/p1, and p02/p01
M2 = sqrt((M1^2 + 2/(gamma-1))/((2*gamma/(gamma-1))*M1^2 - 1));
p_ratio = (2*gamma*M1^2 - (gamma-1))/(gamma+1);
% p0 = p*(1 + (gamma-1)/2*M^2)^(gamma/(gamma-1)) on each side
p0_ratio = p_ratio*((1 + (gamma-1)/2*M2^2)/(1 + (gamma-1)/2*M1^2))^(gamma/(gamma-1));
end
