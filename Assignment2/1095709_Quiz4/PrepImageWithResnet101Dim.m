    function Iout = PrepImageWithResnet101Dim(filename)  
                  
        I = imread(filename);  
        if ismatrix(I)  
            I = cat(3,I,I,I);  
        end  
        %% image resizing to resnet101 dimensions   
        Iout = imresize(I, [224 224]);    
    end  