# Chebfun's standardChop (Aurentz and Trefethen 2017), as checked before adapting it (section 2.3)

Read 2026-11-22 from https://raw.githubusercontent.com/chebfun/chebfun/master/standardChop.m (HTTP 200, 6181 characters),
through a fetch subagent; the function body quoted below is the file's code.

```matlab
if ( tol >= 1 ), cutoff = 1; return, end
n = length(coeffs); cutoff = n; if ( n < 17 ), return, end
% Step 1: envelope
b = abs(coeffs); m = b(end)*ones(n, 1);
for j = n-1:-1:1, m(j) = max(b(j), m(j+1)); end
if ( m(1) == 0 ), cutoff = 1; return, end
envelope = m/m(1);
% Step 2: plateau point
for j = 2:n
    j2 = round(1.25*j + 5);
    if ( j2 > n ), return, end
    e1 = envelope(j); e2 = envelope(j2);
    r = 3*(1 - log(e1)/log(tol));
    plateau = (e1 == 0) | (e2/e1 > r);
    if ( plateau ), plateauPoint = j - 1; break, end
end
% Step 3: cutoff
if ( envelope(plateauPoint) == 0 )
    cutoff = plateauPoint;
else
    j3 = sum(envelope >= tol^(7/6));
    if ( j3 < j2 ), j2 = j3 + 1; envelope(j2) = tol^(7/6); end
    cc = log10(envelope(1:j2)); cc = cc(:);
    cc = cc + linspace(0, (-1/3)*log10(tol), j2)';
    [~, d] = min(cc);
    cutoff = max(d - 1, 1);
end
```

The file's comment: it chops beyond TOL^(2/3) relative to max|coeffs|, never unless n >= 17 and the tail falls below
TOL^(1/3); the constants "result from extensive experimentation" and are "not derived from first principles".
Its tol is a target relative accuracy (default Chebfun eps); a vector it does not chop returns cutoff = n ("not happy").
