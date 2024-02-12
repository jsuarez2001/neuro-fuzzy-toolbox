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


% ------ PREMISES UPDATING ------
suma=zeros(rules,dim);
tempa=zeros(rules,dim);
sumc=zeros(rules,dim);
tempc=zeros(rules,dim);

   
%Cada dato de entrada hace un pequeño aporte a la suma para asi poder
%actualizar los parametros de la premisa.
inputs = 3;
k=0.01;

for l=1:inputs
    for j=1:dim,
        xij=x(l,j);
        for i = 1 : rules
            if pp{j}(i,1)==0
                pp{j}(i,1)=1e-6;
            end
            tempa(i,j)= (-4/(pp{j}(i, 1))^3)*((xij-pp{j}(i, 3))^2)*w_norm(l,i)*(fk(l,i)-f(l))*(y(l,:)-f(l));
            %Actualizacion Parametro c
            tempc(i,j)= (-4/(pp{j}(i, 1))^2)*(xij-pp{j}(i, 3))*w_norm(l,i)*(fk(l,i)-f(l))*(y(l,:)-f(l));
            %actualizacion_parametro(w_norm(l,:),fk(l,:),f(l,:),D.y_train(l,:),D.x_(l,:),l);
        end
    end
    suma=suma+tempa;
    sumc=sumc+tempc;
end

for j = 1 : dim
    for i=1:rules
        if abs(suma(i,j))>eps,
            nablaa(i,j)=k/sqrt(suma(i,j).^2);
            %nablaa(i,j)=0.9/D.n_train;
        else
            nablaa(i,j)=0; 
        end
        if abs(sumc(i,j))>eps,
            nablac(i,j)=k/sqrt(sumc(i,j).^2);
            %nablac(i,j)=0.9/D.n_train;
        else
            nablac(i,j)=0; 
        end
        %nablaa(i)=0.9;nablac(i)=0.9;
    end
end   

freeze=ones(rules,1);

new_pp = pp;
for i = 1 : rules
    if(freeze(i)==1)
        for j = 1 : dim
            new_pp{j}(i, 1)=pp{j}(i, 1)-nablaa(i,j)*suma(i,j);
            new_pp{j}(i, 3)=pp{j}(i, 3)-nablac(i,j)*sumc(i,j);
        end
    end
end
