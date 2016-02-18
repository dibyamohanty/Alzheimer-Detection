%Read Alzheimer dataset
path ='C:\Users\abc\Desktop\math(new.)\MI\Data\1\';
files=dir([path,'*fseg*']);

for i=1:2:length(files)
    j=floor(i/2)+1;
    nii=load_nii(files(i).name);
    fname=[num2str(j) '.nii'];
    save_nii(nii,fname);
    I_orig(j)=load_nii(fname);
    I_thresh(j)=load_nii(fname);
    %view_nii(I_orig(j));
end

flen=length(I_orig);

%Read Cognitive Normal dataset
path1 ='C:\Users\abc\Desktop\math(new.)\MI\Data\2\';
files1=dir([path1,'*fseg*']);

for i=1:2:length(files1)
    j=flen + floor(i/2)+1;
    nii=load_nii(files(i).name);
    fname=[num2str(j) '.nii'];
    save_nii(nii,fname);
    I_orig(j)=load_nii(fname);
    I_thresh(j)=load_nii(fname);
    %view_nii(I_orig(j));
end

%Extract features from Brain MRI
for i=1:1:length(I_orig)
    J=zeros(176,208,176);
    
   
    volg(i)=0;
    tvol(i)=0;
    volw(i)=0;
    volc(i)=0;
    for p=1:176
        for q=1:208
            for r=1:176
                if I_orig(i).img(p,q,r)==2
                    J(p,q,r)=I_orig(i).img(p,q,r);
                    volg(i)=volg(i)+1;
                end
                
                if I_orig(i).img(p,q,r)==(1|2|3)
                        tvol(i)=tvol(i)+1;
                end
                if I_orig(i).img(p,q,r)==1
                    K(p,q,r)=I_orig(i).img(p,q,r);
                    volw(i)=volw(i)+1;
                end
                if I_orig(i).img(p,q,r)==3
                    L(p,q,r)=I_orig(i).img(p,q,r);
                    volc(i)=volw(i)+1;
                end
            end
        end
    end 
    
    %Volume proportion feature
    gm_vol_r(i)=vol(i)./tvol(i);
    wc_vol_r(i)=volw(i)./volc(i);
	for p=1:176
        for q=1:208
            for r=1:176
                I_thresh(i).img(p,q,r)=J(p,q,r);
            end
        end
	end



    %view_nii(I_thresh(i));
    
    feature_vector=zeros(1,6);
    statsum=zeros(1,4);
    fstats=zeros(1,4);

    %GLCM Features
	for p=1:16:176
		I_slice= I_thresh(i).img(:,:,p);
		glcm=graycomatrix(I_slice,'GrayLimits',[0 1],'Offset',[1 0]);
		stats=struct2array(graycoprops(glcm,'all'));
		statsum=cat(1,statsum,stats);
	end
 
	for j=1:12
		fstats(1)=statsum(j)+fstats(1);
	end

	fstats=fstats./11;
	x=[fstats gm_vol_r(i) wc_vol_r(i)];
	feature_vector = cat(1,feature_vector,x);
end

%SVM Classification 5 fold training and testing
feat=feature_vector;
x_pos_train=feat(1:40,:);
y_pos_train=ones(40,1);
x_neg_train=feat(65:76,:);
y_neg_train = zeros(12,1);
x_pos_test = feat(41:64,:);
y_pos_test = ones(24,1);
x_neg_test = feat(77:end,:);
y_neg_test = zeros(8,1);
x_train =cat(1,x_pos_train,x_neg_train);
y_train = cat(1,y_pos_train,y_neg_train);
svm = svmtrain(x_train,y_train);
x_test = cat(1,x_pos_test,x_neg_test);
y_test = cat(1,y_pos_test,y_neg_test);
disp('done')

y_pred = svmclassify(svm,x_test);
acc1 = y_test-y_pred;
correct=0;

%Accuracy and Sensitivity Calculation
for i=1:length(acc1)
   if acc1(i)==0
       correct=correct+1;
   end
end
accuracy = correct/size(acc1,1)

tp=0;tn=0;fp=0;fn=0;
for j=1:length(acc1)
   if y_test(j)==1
        if y_pred(j)==1
            tp=tp+1;
        elseif y_pred(j) == 0
            fn=fn+1;
        end
   else
       if y_pred(j) ==1;
           fp=fp+1;
       elseif y_pred(j)==0;
           tn=tn+1;
       end
   end
end
tp,tn,fp,fn
sens = tp/(tp+fn)
spec=tn/(fp+tn)
