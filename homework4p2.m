r = 2 + (2-3).*rand(100,1);
s = -pi + (pi+pi).*rand(100,1); 

m(:,1) = [0;0]; Sigma(:,:,1) = 0.1*[1 0;0 1] % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
classPriors = [0.35, 0.65]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rb';
for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end

T = array2table(x.'); 
part = cvpartition(T.Var1,'KFold',10);
mdl1 = fitcsvm(part,'T','KernelFunction','gaussian');
mdl = fitcsvm(part,'T','KernelFunction','linear');