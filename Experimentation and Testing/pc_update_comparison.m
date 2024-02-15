x = [
    0.8272, 0.6737;
    0.6561, 0.6231;
    0.5756, 0.3885
];

y = [
    0.5;
    2.0;
    2.5
];


dim = 2;


%rules = 3;

%pp{1} = [
%         0.3922, -0.1463,  0.0462;
%         0.6555,  0.1112,  0.0971;
%         0.1712, -0.6779,  0.8235
%];

%pp{2} = [
%         0.6948, -0.1463,  0.7655;
%         0.3171,  0.1112,  0.7952;
%         0.9502, -0.6779,  0.1869
%];

%pc = [
%    -0.3726, -0.1463,  0.6341;
%    -0.1292,  0.1112,  0.6404;
%     0.6947, -0.6779, -0.4394
%];


rules = 2;

pp{1} = [
         0.3922, -0.1463,  0.0462;
         0.6555,  0.1112,  0.0971
];

pp{2} = [
         0.6948, -0.1463,  0.7655;
         0.3171,  0.1112,  0.7952
];


pc = [
    -0.3726, -0.1463,  0.6341;
    -0.1292,  0.1112,  0.6404
];



%Calculo de f (salidas finales), w_norm (pesos normalizados)
%y fk (salidas por regla)
mu=[];
for el = 1 : size(x, 1)
    % Cálculo de los grados de pertenencia.
    for i = 1 : rules
        for j = 1 : dim
            mu(j,i) = exp(-0.5*((x(el, j)-pp{j}(i, 3))/pp{j}(i, 1))^2);
        end
    end
    w = prod(mu, 1);
    if sum(w)~=0,
        w_norm(el,:) = w./sum(w);
    else
        w_norm(el,:) = w;
    end
end
xExp = [x, ones(size(x, 1),1)]; 
fk=xExp*pc';
f=sum(fk.*w_norm,2);

% ------ CONSEQUENTS UPDATING ------

tmp_pc = pc;
n_datos = 3;
n_dims = dim;

X = [x, ones(n_datos,1)];
X = repmat(X,1,rules);
Y = y;

fs = zeros(n_datos,rules*(n_dims+1));
for k = 1:n_datos;
    fs_tmp = [];
    for i = 1:rules
        fs_tmp = [fs_tmp repmat(w_norm(k,i),1,n_dims+1)];
    end
    fs(k,:) = fs_tmp;
end

disp('X');
disp(X);
disp('fs');
disp(fs);
X = X.*fs;
disp('X.*fs');
disp(X);
disp('Y');
disp(Y);
consec_tmp = regress(Y,X);

disp('consec_tmp');
disp(consec_tmp);
pc = reshape(consec_tmp,n_dims+1,rules)';
disp('pc');
disp(pc);


freeze=ones(rules,1);

for r = 1:rules
    if freeze(r)==0
        pc(r,:) = tmp_pc(r,:);
    end
end

newfk=xExp*pc';
newf=sum(newfk.*w_norm,2);
