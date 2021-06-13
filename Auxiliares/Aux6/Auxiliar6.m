%% Auxiliar 6 Econometria - IN709 Otono 2021
clc
clear all

%% Preliminares
rng(709);
n=1000;
B_pob=[10 -1 -1]';
v=[2 -0.1;-0.1 1]; % matriz varianzas covarianzas de la distribución de X

X23=mvnrnd([0 0]',v,n);
X=[ones(n,1) X23]; % matriz de datos dimension nxk
k=size(X,2); % size(X)=[1000 3]
%% Heterocedasticidad

%% Shock es una realizacion de normal de media 0 y varianza (X_1i^2+X_2i^2)/2)
% Generamos los errores con dicha especificacion.

% Rescatar 2a y 3a columna
a2=X(:,2);
b2=X(:,3);

e_het=zeros(n,1);
for i=1:n
    e_het(i)=(((a2(i)^2+b2(i)^2)/2)^0.5)*randn; %randn: escalar -> N(0,1), ponderado por "A^{0.5}", N(0,A).
end

%Modelo con Heterocedasticidad
Y=X*B_pob+e_het;

%% 1) MCO y Varianza A
B=(X'*X)^-1*X'*Y;
e_ols=(Y-X*B);
var_A= (e_ols'*e_ols)/(n-k)*(X'*X)^-1;
% Recordar que s^2=(e_ols'*e_ols)/(n-k)

%% 2) LM Test
%paso 1: hecho en la parte anterior
%paso 2: se considera la regresion basica, los z a usar seran las variables
%del modelo al cuadrado, estas pueden ser otras potencias, o
%multiplicaciones entre variables
e_ols2=e_ols.^2;
Z=[ones(n,1) X(:,2:3).^2];

delta_hat=(Z'*Z)^(-1)*Z'*e_ols2; %estimadores regresion basica
r_hat=e_ols2-Z*delta_hat; %residuos regresion basica
media_y=mean(e_ols2);
R_cuadrado=1-(r_hat'*r_hat)/((e_ols2-media_y)'*(e_ols2-media_y));

% paso 3: calculamos estadistico
estadistico=n*R_cuadrado;

%paso 4: comparamos con valor critico
estadistico<chi2inv(0.95,2);
%rechazamos hipotesis nula, es decir, rechazamos homocedasticidad


%% 3) Eicker-White y Varianza B
%var_gls=((X'*X)^-1*estimador*(X'*X)^-1);
%Estimador sum(e_i^2*x_i'*x_i)
estimador=X'*diag((Y-X*B).^2)*X;
var_B=((X'*X))^-1*estimador*((X'*X))^-1;

%% 4) Minimos Cuadrados Factibles 1: Varianza parcial (erronea)
var_des=(X(:,2).^2/2);
omega=diag(var_des);
%omega es sigma2*OMEGA
B_fgls=(X'*omega^-1*X)^-1*X'*omega^-1*Y;
e_fgls=Y-X*B_fgls;
var_e_fgls=(e_fgls'*e_fgls)/(n-k);
var_C=var_e_fgls*(X'*omega^-1*X)^-1;

%% 5) Minimos Cuadrados Factibles 1: Varianza conocida
var_con=(0.5*(a2.^2+b2.^2));
omega2=diag(var_con);
%omega es sigma2*OMEGA
B_fgls2=(X'*omega2^-1*X)^-1*X'*omega2^-1*Y;
e_fgls2=Y-X*B_fgls2;
var_e_fgls2=(e_fgls2'*e_fgls2)/(n-k);
var_D=var_e_fgls2*(X'*omega2^-1*X)^-1;

%% 6) Rankear formalmente las matrices A, B y D en terminos de eficiencia

% Corregir varianza de OLS en presencia de heterocedasticidad
var_A= (e_ols'*e_ols)/(n-k)*(X'*X)^-1*(X'*omega2*X)*(X'*X)^-1 %Var OLS cuando hay heterocedasticidad

h1=eig(var_A-var_B) %Todos positivos, A mayor varianza que B.
h2=eig(var_A-var_D) %Todos positivos, A mayor varianza que D.
h3=eig(var_B-var_D) %Todos positivos, B mayor varianza que D.
