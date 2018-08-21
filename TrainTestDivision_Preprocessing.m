%% Preprocessing and traning-testing division

    %%%%    Authors:    DIBYASUNDAR DAS AND DEEPAK RANJAN NAYAK 
    %%%%    NATIONAL INSTITUTE OF TECHNOLOGY ROURKELA, INDIA
    %%%%    EMAIL:      DIBYASUNDARIT@GMAIL.COM; DEPAKRANJANNAYAK@GMAIL.COM
    %%%%    WEBSITE:    https://www.researchgate.net/profile/Deepak_Nayak11
    %%%%    DATE:       AUGUST 2018

clc;clear;close all;
    req_m=100;
    req_n=400;
    if ispc % To run in Windows environment
        list_dir=dir('Database\');% list all directory in Database (includes ..\ and .\) so real folders starts from 3
        folder_name='Database\'; % Database contains word images of 120 classes
    elseif isunix % To run in Linux environment
        list_dir=dir('Database/');
        folder_name='Database/';
    end
    class_data={};
    for i=3:size(list_dir,1)
        list_file=[];
        if ispc
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'\*.jpg'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'\*.bmp'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'\*.tif'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'\*.png'))];
        elseif isunix
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'/*.jpg'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'/*.bmp'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'/*.tif'))];
            list_file=[list_file;dir(strcat(folder_name,list_dir(i).name,'/*.png'))];
        end
        t_data={};
        for j=1:size(list_file,1)
            if ispc
                img_path=strcat(list_file(j).folder,'\',list_file(j).name);
            elseif isunix
                img_path=strcat(list_file(j).folder,'/',list_file(j).name);
            end
            %% Reading image
            img=imread(img_path);
            [m,n,p]=size(img);
            %% Preprocessing
            while m>=req_m || n>=req_n
                img=imresize(img,0.5,'bilinear');
                [m,n,p]=size(img);
            end
            if p==3
                img=1-im2double(rgb2gray(img));
            else
                img=1-im2double(img);
            end
            template=zeros(req_m,req_n);
            x=int32(req_m/2-m/2);
            y=int32(req_n/2-n/2);
            template(x:x+m-1,y:y+n-1)=img;
            t_data{j}=template;
        end
        %% Saving preprocessed image classwise in .mat file 
        class_data=t_data;
        save(strcat('class_data_',num2str(i-2),'.mat'),'class_data')
    end
    %% Traning Testing Division 
    clear;clc;
    train_data=[];
    train_cls=[];
    test_data=[];
    test_cls=[];
    kk=1;kl=1;
    file_list=dir('*.mat');
    for i=1:120
        load(strcat('class_data_',num2str(i),'.mat'));
        [m,n]=size(class_data);
        % disp(strcat(num2str(i),':',num2str(n)))
        %% Random division
        pos=randperm(n);
        for j=1:120
            train_data(:,:,1,kk)=class_data{pos(j)};
            train_cls(kk,1)=i;
            kk=kk+1;
        end
        for j=121:150
            test_data(:,:,1,kl)=class_data{pos(j)};
            test_cls(kl,1)=i;
            kl=kl+1;
        end
    end
    save('train_data.mat','train_data','train_cls','-v7.3');
    save('test_data.mat','test_data','test_cls','-v7.3');