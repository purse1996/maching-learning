clear ; 
close all;
clc

load('pingguo.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_apple_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_apple_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('yingtao.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_yingtao_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_yingtao_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('putao.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_putao_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_putao_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('mihoutao.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_mihoutao_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_mihoutao_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('ningmeng.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_ningmeng_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_ningmeng_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('caomei.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_caomei_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_caomei_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('chengzi.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_chengzi_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_chengzi_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('taozi.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_taozi_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_taozi_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('li.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_li_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_li_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('shiliu.mat')
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model_shiliu_linear = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
fprintf('\nchoose the bset parameters ...\n')
[C, sigma] = dataset3Params(X, y, Xval, yval);
fprintf('\nEvaluating the Gaussian Kernel ...\n')
model_shiliu_gaussian= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

load('test.mat');
x_test = X_test(1,:);
y1 = svmPredict(model_apple_linear,x_test);
y2 = svmPredict(model_yingtao_linear,x_test);
y3 = svmPredict(model_putao_linear,x_test);
y4 = svmPredict(model_mihoutao_linear,x_test);
y5 = svmPredict(model_ningmeng_linear,x_test);
y6 = svmPredict(model_caomei_linear,x_test);
y7 = svmPredict(model_chengzi_linear,x_test);
y8 = svmPredict(model_taozi_linear,x_test);
y9 = svmPredict(model_li_linear,x_test);
y10 = svmPredict(model_shiliu_linear,x_test);

pred = [y1 y2 y3 y4 y5 y6 y7 y8 y9 y10];
[value,position] = max(pred);
 switch pred
        case 1
            set(handles.edit3,'string','Æ»¹û');
        case 2
            set(handles.edit3,'string','Ó£ÌÒ');
        case 3
            set(handles.edit3,'string','ÆÏÌÑ');
        case 4
            set(handles.edit3,'string','â¨ºïÌÒ');
        case 5
            set(handles.edit3,'string','ÄûÃÊ');
        case 6
            set(handles.edit3,'string','²ÝÝ® ');
        case 7
            set(handles.edit3,'string','³È×Ó');
        case 8
            set(handles.edit3,'string','ÌÒ×Ó');
        case 9
            set(handles.edit3,'string','Àæ');
        case 10
            set(handles.edit3,'string','Ê¯Áñ');            
    end
end

 