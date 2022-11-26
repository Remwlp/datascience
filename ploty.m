function ploty
clc;clear;

filename1 = 'sample.txt';
filename2 = 'sp.txt';
[data1,data2]=textread(filename1,'%n%n','delimiter', ',');
[data3,data4]=textread(filename2,'%n%n','delimiter',',');
figure
hold on;box on;
axis([114.2,114.65,30.45,30.7])
plot(data1,data2,'bo','MarkerFaceColor','b');
plot(data3,data4,'-r+');

legend('ÂË²¨¹ì¼£','Ñ¹Ëõ¹ì¼£');