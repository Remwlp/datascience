%  Kalman�˲��ڴ���GPS������λϵͳ�е�Ӧ��
function kalman
clc;clear;
filename = 'E:\ѧϰ����\����\ѡ��\making\20161025 (22).txt';
[data1,data2,data3,data4]=textread(filename,'%n%n%n%n','delimiter', ',');
speed1=data3.*sind(data4);
speed2=data3.*cosd(data4);
fid = fopen(filename);
lines = 0;
while ~feof(fid)
    fgetl(fid);
    lines = lines +1;
end

N=lines;%�ܵĲ�������
X=zeros(4,N);%Ŀ����ʵλ�á��ٶ�
X=[data1,speed1,data2,speed2]';
Z=zeros(2,N);%��������λ�õĹ۲�
Z=[data1,data2]';

for i=1:2
    for j=2:657
        if abs(Z(i,j)-Z(i,j-1))>0.05  %���˵���������
            Z(:,j)=[];
            N=N-1;
        else
            Z(i,j)=(Z(i,j)+Z(i,j-1))/2;
        end
    end
end


epiod=1e-6;
Q=epiod*diag([0.5,1,0.5,1]) ;%����������ֵ
R=0.00001*eye(2);
F=[1,1,0,0;0,1,0,0;0,0,1,1;0,0,0,1];%״̬ת�ƾ���
H=[1,0,0,0;0,0,1,0];%�۲����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�������˲�
Xkf=zeros(4,N);
Xkf(:,1)=X(:,1);%�������˲�״̬��ʼ��
P0=eye(4);%Э�������ʼ��
for i=2:N
    Xn=F*Xkf(:,i-1);%Ԥ��
    P1=F*P0*F'+Q;%Ԥ�����Э����
    K=P1*H'*inv(H*P1*H'+R);%����
    Xkf(:,i)=Xn+K*(Z(:,i)-H*Xn);%״̬����
    P0=(eye(4)-K*H)*P1;%�˲����Э�������
end


%������
for i=1:N
    Err_Observation(i)=RMS(X(:,i),Z(:,i));%�˲�ǰ�����
    Err_KalmanFilter(i)=RMS(X(:,i),Xkf(:,i));%�˲�������
end
%��ͼ
fclose(fid);
figure
hold on;box on;
axis([114.2,114.65,30.45,30.7])
plot(data1,data2,'-.o');
plot(Xkf(1,:),Xkf(3,:),'-r.');%�������˲��켣
legend('�۲�켣','�˲��켣');
flie=fopen('E:\ѧϰ����\����\ѡ��\sample.txt','w');
for i=1:657
    fprintf(flie,'%g,%g\n',Xkf(1,i),Xkf(3,i));
end
figure
hold on; box on;
axis([-inf,inf,0,0.01])
plot(Err_KalmanFilter,'-ks','MarkerFace','r')
legend('�˲����')
fclose('all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ŷʽ�����Ӻ���
function dist=RMS(X1,X2)
if length(X2)<=2
    dist=sqrt( (X1(1)-X2(1))^2 + (X1(3)-X2(2))^2 );
else
    dist=sqrt( (X1(1)-X2(1))^2 + (X1(3)-X2(3))^2 );
end
%%%%%%%%%%%%%%%%%%%%%%%%
