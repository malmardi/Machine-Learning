load('C:\Users\malmardi\Dropbox\machinelearning\HW3\ITCS6156_HW3 (1)\ITCS6156_HW3\optdigits_raining.csv');

load('C:\Users\malmardi\Dropbox\machinelearning\HW3\ITCS6156_HW3 (1)\ITCS6156_HW3\optdigits_test.csv');

temp_train=optdigits_raining(:,1:64);
temp_train_labels=optdigits_raining(:,65);
temp_test=optdigits_test(:,1:64);
temp_test_labels=optdigits_test(:,65);


array_res=zeros(size(temp_train_labels,1),10);
for i=1:size(temp_train_labels,1)
   col=temp_train_labels(i,1)+1;
    array_res(i,col)=1;
end

array_res2=zeros(size(temp_test_labels,1),10);
for i=1:size(temp_test_labels,1)
   col=temp_test_labels(i,1)+1;
    array_res2(i,col)=1;
end


train_data = temp_train';
train_labels = array_res';

 test_data= temp_test';
 test_labels=array_res2';
 
 
correct = zeros(20,1);
accuracy = zeros(25,1);
neurons_set = [11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49];
for n=1:size(neurons_set,2)
neuron = neurons_set(n);
layersizes = [neuron];
net = fitnet(layersizes);
net.trainParam.epochs = 75;
net = train(net,train_data,train_labels);

%testing code
correct(neuron) = 0;
out=net(test_data);
for i=1:size(test_labels,2)
target = test_labels(:,i);

[argvaluey, argmax] = max(out(:,i));
x=(argmax);

%y = argmax(target);
[argvalue, argmaxy] = max(target);
y=(argmaxy);


if (x==y)
   correct(neuron) = correct(neuron)+1;
end
end
accuracy(neuron)=(correct(neuron)/size(test_data, 2))*100;
fprintf(1,'\nperct correct =%d\n',((correct(neuron)/size(test_data, 2))*100));
end


accuracy2 = zeros(25,1);
index=1;
for i=1:size(accuracy)
    if(accuracy(i,1)>0)
        accuracy2(index,1)=accuracy(i,1);
        index=index+1;
    end
end

plot(neurons_set, accuracy2) 
 