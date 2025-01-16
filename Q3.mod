
%--------------------------------------------------------------------------
                            % Endogenous Varibales
%--------------------------------------------------------------------------

var zz              ${z}$                   (long_name = 'technology process')
    yy              ${y}$                   (long_name = 'output')
    cc              ${c}$                   (long_name = 'consumption')
    kk              ${k}$                   (long_name = 'capital')
    ll              ${l}$                   (long_name = 'labor')
    rk              ${r^k}$                 (long_name = 'return on capital')
;

%--------------------------------------------------------------------------
                            % Exogenous Variables
%--------------------------------------------------------------------------

varexo  eepsilon    ${\varepsilon}$         (long_name = 'technology process shock')
    
;

%--------------------------------------------------------------------------
                            % Declaration of Parameters
%--------------------------------------------------------------------------
parameters

    bbeta           ${\beta}$               (long_name = 'discount factor')
    ddelta          ${\delta}$              (long_name = 'depreciation rate')
    aalpha          ${\alpha}$              (long_name = 'elasticity of output wrt capital')
    rrho            ${\rho}$                (long_name = 'persistence of technological process')
    ssigma          ${\sigma}$              (long_name = 'standard deviation of technological process')
    k_to_l          ${k/l}$                 (long_name = 'capital-to-labor ratio')
    c_to_l          ${c/l}$                 (long_name = 'consumption-to-labor ratio')

;

%--------------------------------------------------------------------------
                        %PARAMETRIZATION
%--------------------------------------------------------------------------

bbeta = 0.97;
ddelta = 0.1;
aalpha = 0.33;
rrho = 0.95;
ssigma = 0.007;

%--------------------------------------------------------------------------
                             %MODEL
%--------------------------------------------------------------------------
model;

[name='Labor-Consumption Choice']
exp(cc) = (1-aalpha)*exp(zz)*exp(kk(-1))^aalpha*exp(ll)^(-1-aalpha);

[name='Euler Equation']
1/exp(cc)=bbeta*(1/exp(cc(+1)))*(aalpha*exp(zz(+1))*exp(kk)^(aalpha-1)*exp(ll(+1))^(1-aalpha) + 1 - ddelta);

[name='Market clearing condition']
exp(cc) + exp(kk) - (1-ddelta)*exp(kk(-1)) = exp(yy);

[name='Output definition']
exp(yy) = exp(zz)*exp(kk(-1))^aalpha*exp(ll)^(1-aalpha);

[name='Technology process']
zz = rrho*zz(-1) + ssigma*eepsilon;

[name='Definition of return on capital']
exp(rk) = aalpha*exp(yy-kk(-1)) - ddelta;

end;

%--------------------------------------------------------------------------
                             %STEADY STATE
%--------------------------------------------------------------------------
steady_state_model;

zz = 0;

k_to_l = ((1/aalpha)*(1/bbeta - 1 + ddelta))^(1/(aalpha-1));
c_to_l = k_to_l^aalpha - ddelta*k_to_l;
ll = 0.5*log((1-aalpha)*(k_to_l^aalpha)/c_to_l);
cc = log(c_to_l) + ll;
kk = log(k_to_l) + ll;
yy = log(exp(zz)*exp(kk)^aalpha*exp(ll)^(1-aalpha));
rk = log(aalpha*exp(yy-kk) - ddelta);

end;


steady;
check;

shocks;

var eepsilon; stderr 1;

end;

stoch_simul(order=3,periods=10000,irf=100,pruning) zz ll kk yy cc rk;
