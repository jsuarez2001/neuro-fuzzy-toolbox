x = [
    0.6294    0.2647    0.9150    0.9143   -0.1565;
    0.8116   -0.8049    0.9298   -0.0292    0.8315;
   -0.7460   -0.4430   -0.6848    0.6006    0.5844;
    0.8268    0.0938    0.9412   -0.7162    0.9190
];

pc = [
    0.6557, 0.9340, 0.7431, 0.1712, 0.2769, 0.8235;
    0.0357, 0.6787, 0.3922, 0.7060, 0.0462, 0.6948;
    0.8491, 0.7577, 0.6555, 0.0318, 0.0971, 0.3171
];

dim = 5;
rules = 3;

m = minmax(x')';
ex = m(2, :) - m(1, :);
stp = ex ./ (rules-1);

for i = 1 : dim
    h=m(1,i)':stp(i):m(2,i)';
    for j = 1 : rules
        pp{i}(j, 1) = stp(i)/2;
        pp{i}(j, 2) = 2;
        pp{i}(j, 3) = h(j);
    end
end

mu=[];
for el = 1 : size(x, 1)
    
    % Cálculo de los grados de pertenencia.
    for i = 1 : 3
        for j = 1 : 5
            mu(j,i) = exp(-((x(el, j)-pp{j}(i, 3))/pp{j}(i, 1))^2);
        end
    end

    % Normalización de los wi.
    w = prod(mu, 1);


    if sum(w)~=0,
        w_norm(el,:) = w./sum(w);
    else
        w_norm(el,:) = w;
    end
end

% Cálculo de las salidas.
    xExp = [x, ones(size(x, 1),1)];
    
    % fn = salida de cada bloque después del consecuente.
    %for k=1:dataANFIS.numMF,
    %    fk(l,k) = (dataANFIS.pc(k,:) * xExp)';
    %end
    
    fk=xExp*pc';
    fn=sum(fk.*w_norm,2);