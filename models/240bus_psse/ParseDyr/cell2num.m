function A=cell2num(D,n)
[h,l]=size(D);
A=zeros(h,l);
for i=1:h
    for j=1:l
        index=find(n==j);
        if index~=0
            A(i,j)=0;
            continue
        end
        k = findstr('/',cell2mat(D(i,j)));
        if length(k)==0
            A(i,j) = str2num(cell2mat(D(i,j)));
        else
            temp=cell2mat(D(i,j));
            A(i,j) = str2num(temp(1:(k-1)));
            return
        end
    end
end
end