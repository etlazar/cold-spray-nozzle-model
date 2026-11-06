function property = find_property(prop,val0,var,constants)
    switch prop
        case 'temp'
            property = val0./(1+(constants(1)-1)/2.*var.^2);
        case 'pressure'
            property = val0./((1+(constants(1)-1)/2*var.^2).^(constants(1)/(constants(1)-1)));
        case 'density'
            property = val0./((1+(constants(1)-1)/2*var.^2).^(1/(constants(1)-1)));
        case 'velocity'
            temp = val0./(1+(constants(1)-1)/2*var.^2);
            property = var .* sqrt(constants(1).*constants(2).*temp);
        case 'viscosity'
            property = val0 .* ((constants(1) + constants(2)) ./ (var + constants(2))) .* (var ./ constants(1)).^(3/2);
        case 'kg'
            property = 1e-4 * exp(0.94306384*log(var) + 0.12279898e3./var ...
                          - 0.11839435e5./var.^2 - 0.10668773);
        case 'Pr'
            property = (constants .* var(1,:))./var(2,:);
        case 'Nu'
            property = 2 + 0.6 .* var(1,:).^0.5 .* var(2,:).^(1/3);
        case 'hp'
            property = var(1,:) .* (var(2,:) ./ constants);
        otherwise
            error('Unknown property: %s', prop);

    end
           
end