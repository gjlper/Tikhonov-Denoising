function [I1 I1_noise I2 I2_noise] = LoadData

I1 = zeros(1000,1);
I1(1:200) = 1;
I1(201:300) = 4;
I1(301:600) = 2;
I1(601:650) = 3;
I1(651:end) = 1;
I1_noise = I1 + 0.2*randn(size(I1));
% figure(1);
% plot(I1,'r-','LineWidth',2);
% hold on;
% plot(I1_noise,'r.');
% hlen = legend('Clean','Noisy');
% set(hlen,'FontSize',20);
% set(gca,'FontSize',2);


I2 = imread('data/Lena.bmp');
I2 = double(I2);
r = 20;
I2_noise = I2 + r*randn(size(I2));
% figure(2); 
% subplot(1,2,1);
% imshow(I2,[]);
% title('Clean image','FontSize',20);
% subplot(1,2,2);
% imshow(I2_noise,[]);
% title('Noisy image','FontSize',20);