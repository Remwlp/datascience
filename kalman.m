%  Kalman滤波在船舶GPS导航定位系统中的应用
function kalman
clc;clear;
filename = 'E:\学习资料\国赛\选拔\making\20161025 (22).txt';
[data1,data2,data3,data4]=textread(filename,'%n%n%n%n','delimiter', ',');
speed1=data3.*sind(data4);
speed2=data3.*cosd(data4);
fid = fopen(filename);
lines = 0;
while ~feof(fid)
    fgetl(fid);
    lines = lines +1;
end

N=lines;%总的采样次数
X=zeros(4,N);%目标真实位置、速度
X=[data1,speed1,data2,speed2]';
Z=zeros(2,N);%传感器对位置的观测
Z=[data1,data2]';

for i=1:2
    for j=2:657
        if abs(Z(i,j)-Z(i,j-1))>0.05  %过滤掉紊乱数据
            Z(:,j)=[];
            N=N-1;
        else
            Z(i,j)=(Z(i,j)+Z(i,j-1))/2;
        end
    end
end


epiod=1e-6;
Q=epiod*diag([0.5,1,0.5,1]) ;%过程噪声均值
R=0.00001*eye(2);
F=[1,1,0,0;0,1,0,0;0,0,1,1;0,0,0,1];%状态转移矩阵
H=[1,0,0,0;0,0,1,0];%观测矩阵
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%卡尔曼滤波
Xkf=zeros(4,N);
Xkf(:,1)=X(:,1);%卡尔曼滤波状态初始化
P0=eye(4);%协方差阵初始化
for i=2:N
    Xn=F*Xkf(:,i-1);%预测
    P1=F*P0*F'+Q;%预测误差协方差
    K=P1*H'*inv(H*P1*H'+R);%增益
    Xkf(:,i)=Xn+K*(Z(:,i)-H*Xn);%状态更新
    P0=(eye(4)-K*H)*P1;%滤波误差协方差更新
end


%误差更新
for i=1:N
    Err_Observation(i)=RMS(X(:,i),Z(:,i));%滤波前的误差
    Err_KalmanFilter(i)=RMS(X(:,i),Xkf(:,i));%滤波后的误差
end
%画图
fclose(fid);
figure
hold on;box on;
axis([114.2,114.65,30.45,30.7])
plot(data1,data2,'-.o');
plot(Xkf(1,:),Xkf(3,:),'-r.');%卡尔曼滤波轨迹
legend('观测轨迹','滤波轨迹');
flie=fopen('E:\学习资料\国赛\选拔\sample.txt','w');
for i=1:657
    fprintf(flie,'%g,%g\n',Xkf(1,i),Xkf(3,i));
end
figure
hold on; box on;
axis([-inf,inf,0,0.01])
plot(Err_KalmanFilter,'-ks','MarkerFace','r')
legend('滤波误差')
fclose('all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算欧式距离子函数
function dist=RMS(X1,X2)
if length(X2)<=2
    dist=sqrt( (X1(1)-X2(1))^2 + (X1(3)-X2(2))^2 );
else
    dist=sqrt( (X1(1)-X2(1))^2 + (X1(3)-X2(3))^2 );
end
%%%%%%%%%%%%%%%%%%%%%%%%
