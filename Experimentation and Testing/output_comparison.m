x = [
    0.8272, 0.6737;
    0.6561, 0.6231;
    0.5756, 0.3885
];

dim = 2;
rules = 3;

pp{1} = [
         0.3922, -0.1463,  0.0462;
         0.6555,  0.1112,  0.0971;
         0.1712, -0.6779,  0.8235
];

pp{2} = [
         0.6948, -0.1463,  0.7655;
         0.3171,  0.1112,  0.7952;
         0.9502, -0.6779,  0.1869
];


pc = [
    -0.3726, -0.1463,  0.6341;
    -0.1292,  0.1112,  0.6404;
     0.6947, -0.6779, -0.4394
];



%m = minmax(x')';
%ex = m(2, :) - m(1, :);
%stp = ex ./ (rules-1);

%for i = 1 : dim
%    h=m(1,i)':stp(i):m(2,i)';
%    for j = 1 : rules
%        pp{i}(j, 1) = stp(i)/2;
%        pp{i}(j, 2) = 2;
%        pp{i}(j, 3) = h(j);
%    end
%end

mu=[];
for el = 1 : size(x, 1)
    %disp('x');
    %disp(el);
    %disp('-');
    
    % Cálculo de los grados de pertenencia.
    for i = 1 : rules
        for j = 1 : dim
            mu(j,i) = exp(-0.5*((x(el, j)-pp{j}(i, 3))/pp{j}(i, 1))^2);
        end
    end

    disp('mu');
    disp(mu);

    % Normalización de los wi.
    w = prod(mu, 1);

    %disp('w');
    %disp(w);


    if sum(w)~=0,
        w_norm(el,:) = w./sum(w);
    else
        w_norm(el,:) = w;
    end
end

disp('w_norm');
disp(w_norm);

% Cálculo de las salidas.
    xExp = [x, ones(size(x, 1),1)];
    
    % fn = salida de cada bloque después del consecuente.
    %for k=1:dataANFIS.numMF,
    %    fk(l,k) = (dataANFIS.pc(k,:) * xExp)';
    %end
    
    fk=xExp*pc';
    fn=sum(fk.*w_norm,2);