%%This program can be used for conversion of images from a dataset to original images
%Used this algo to convet CIFAR100/10 datasets into original images

if true
  clc;
  clear all;
  load('meta.mat');
  %Create folders
  for i=1:length(fine_label_names)
     mkdir('CIFAR-100\TEST\',fine_label_names{i});
     mkdir('CIFAR-100\TRAIN\',fine_label_names{i});
  end
  %%Training images
  load('train.mat');
  im=zeros(32,32,3);
  for cpt=1:50000   
     R=data(cpt,1:1024);
     G=data(cpt,1025:2048);
     B=data(cpt,2049:3072);
     k=1;
     for x=1:32
        for i=1:32
          im(x,i,1)=R(k);
          im(x,i,2)=G(k);
          im(x,i,3)=B(k);
          k=k+1;
        end
     end  
     im=uint8(im);
     pathdest = strcat('CIFAR-100\TRAIN\',fine_label_names{fine_labels(cpt)+1},'\',filenames{cpt});
     imwrite(im,pathdest,'png'); 
 end
 %%Test images
 load('test.mat');
 im=zeros(32,32,3);
 for cpt=1:10000   
    R=data(cpt,1:1024);
    G=data(cpt,1025:2048);
    B=data(cpt,2049:3072);
    k=1;
    for x=1:32
       for i=1:32
          im(x,i,1)=R(k);
          im(x,i,2)=G(k);
          im(x,i,3)=B(k);
          k=k+1;
       end
    end  
    im=uint8(im);
    pathdest = strcat('CIFAR-100\TEST\',fine_label_names{fine_labels(cpt)+1},'\',filenames{cpt});
    imwrite(im,pathdest,'png'); 
  end
end