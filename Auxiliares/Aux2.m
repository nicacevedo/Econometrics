%Aux2.m
%IN709-Econometría/2021
%%
%limpiar variables y pantalla
clear,clc,
%establecer el generador de números aleatorios
rng('default');

N=10000;% Número de observaciones
K=2; % Dimensión de X (número de regresores)

% Parámetros verdaderos (poblacionales)
beta_true=[1; 3;-2];
sigma=1;

% Media de las variables explicativas
mediaX=5;
% Simular los errores poblacionales
e=0+sigma*randn(N,1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PREGUNTA1: Variables explicativas con menor y mayor variación
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sd1=0.001;%desviación estándar para el caso con baja variación
sd2=1;%desviación estándar para el caso con alta variación
%simular los datos
X1=[ones(N,1),mediaX+ (sd2*randn(N,K))];
X2=[ones(N,1),mediaX+ (sd1*randn(N,K))];
%Comprobar la media y desviación estándar de los datos generados
E_X1=mean(X1);
Sd_X1=sqrt(var(X1));
E_X2=mean(X2);
Sd_X2=sqrt(var(X2));
%Comprobar que no hay correlación en los datos generados(debería
%aproximarse a cero)
correlacion_X1=[corr(X1(:,2),X1(:,3)) corr(X1(:,2),e) corr(X1(:,3),e)];
correlacion_X2=[corr(X2(:,2),X2(:,3)) corr(X2(:,2),e) corr(X2(:,3),e)];
%Obtener las variables dependientes asociadas
Y1=X1*beta_true+e;
Y2=X2*beta_true+e;
%Estimar por MCO
beta1_MCO=(X1'*X1)^-1*X1'*Y1;
beta2_MCO=(X2'*X2)^-1*X2'*Y2;
%Calcular los errores estimados
e1=Y1-(X1*beta1_MCO);
e2=Y2-(X2*beta2_MCO);
%Calcular la varianza de los estimadores
Varbeta1_MCO=diag((e1'*e1/(N-(K+1)))*(X1'*X1)^-1);
Varbeta2_MCO=diag((e2'*e2/(N-(K+1)))*(X2'*X2)^-1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PREGUNTA2: variables explicativas correlacionadas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%definimos la correlación a generar
co=0.95;
%simular datos correlacionados(usamos propiedades de la normal para
%mantener la desviación estándar y la media requeridas)
X3=zeros(N,3);
X3(:,1)=ones(N,1);
X3(:,2)=mediaX+ (sd2*randn(N,1));

%generacion de vectores para formar vector correlacionado
Base1=(X3(:,2)-mediaX)/sd2;
Base2=randn(N,1);
Base3=co*Base1+sqrt(1-co^2)*Base2;
X3(:,3)=mediaX+sd2*Base3;

%obtener correlaciones de los datos generados
correlacion_X3=[corr(X3(:,2),X3(:,3)) corr(X3(:,2),e) corr(X3(:,3),e)];
%Comprobar la media y desviación estándar de los datos generados
E_X3=mean(X3);
Sd_X3=sqrt(var(X3));
%simular la variable dependiente para X3
Y3=X3*beta_true+e;
%Estimar por MCO
beta3_MCO=(X3'*X3)^-1*X3'*Y3;
%Calcular los errores estimados
e3=Y3-(X3*beta3_MCO);
%Calcular la varianza del estimador
Varbeta3_MCO=diag((e3'*e3/(N-(K+1)))*(X3'*X3)^-1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PREGUNTA3: variable explicativa correlacionada con el error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%definimos la correlación a generar
co2=0.95;
%Generar datos correlacionados con el error(usamos las propiedades de la
%normal para mantener la media y desviación estándar)
X4=zeros(N,3);
X4(:,1)=ones(N,1);
X4(:,2)=mediaX+ (sd2*randn(N,1));

%generacion de vectores para formar vector correlacionado
Base14=(e);
Base24=randn(N,1);
Base34=co*Base14+sqrt(1-co^2)*Base24;
X4(:,3)=mediaX+sd2*Base34;




%obtener correlaciones de los datos generados
correlacion_X4=[corr(X4(:,2),X4(:,3)) corr(X4(:,2),e) corr(X4(:,3),e)];
%Comprobar la media y desviación estándar de los datos generados
E_X4=mean(X4);
Sd_X4=sqrt(var(X4));
%simular la variable dependiente para X4
Y4=X4*beta_true+e;
%Estimar por MCO
beta4_MCO=(X4'*X4)^-1*X4'*Y4;
%Calcular los errores estimados
e4=Y4-(X4*beta4_MCO);
%Calcular la varianza del estimador
Varbeta4_MCO=diag((e4'*e4/(N-(K+1)))*(X4'*X4)^-1);
%Calcular sesgo cuando hay una variable correlacionada con el error
sesgo_corr=(X4'*X4)^-1*X4'*e;
%Calcular sesgo con el modelo sin correlación con el error
sesgo_ncorr=(X1'*X1)^-1*X1'*e;