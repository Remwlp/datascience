
clc;clear
filename1 = 'sample.txt';
[data1,data2]=textread(filename1,'%n%n','delimiter', ',');
Points=[data1,data2]';
figure;
ii = 1;
subplot(2,3,ii);hold on
plot(Points(1,:),Points(2,:),'-or');
title('initial points');
grid on;box on;axis([1 8 0.5 3.5]);

for epsilon = [0.0001:0.0001:0.0003 0.0005 0.0025]
    res = DouglasPeucker(Points,epsilon);
    ii = ii + 1;
    subplot(2,3,ii);hold on
    plot(Points(1,:),Points(2,:),'--ob');
    plot(res(1,:),res(2,:),'-xr');
    a=size(res(1,:));
    title(['\epsilon = ' num2str(epsilon)]);
    grid on;box on;
    legend('ÂË²¨¹ì¼£','Ñ¹Ëõ¹ì¼£');
end

