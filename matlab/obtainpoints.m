function [x,y,Iout]=obtainpoints(cntrd,I,cntrd_offset,searchradius)
%% Obtain points along the laser line in wire grid relative angle calibration 
    Iout=zeros(size(I));
    cart=[];
    
for i=1:searchradius  
    top=I(floor(cntrd-cntrd_offset-i),:);
    bot=I(floor(cntrd+cntrd_offset+i),:);
    
    if (sum(top) ~= 0) && (sum(bot) ~= 0)    
    top_weights=repmat(1:size(top,2),1).*top;
    top_cntrd=sum(top_weights)./sum(top);

    out(1)=cntrd-cntrd_offset-i;
    out(2)=top_cntrd;
% %     
%     out(2)=cntrd(2)-cntrd_offset-i;
%     out(1)=top_cntrd;
%     
    cart=[cart;out];
    
%    Iout(cntrd-(cntrd_offset+i),floor(top_cntrd))=1;
    
    
    bot_weights=repmat(1:size(bot,2),1).*bot;
    bot_cntrd=sum(bot_weights)./sum(bot);
    
 %   Iout(cntrd+cntrd_offset+i,floor(bot_cntrd))=1;
%     
    out(1)=cntrd+cntrd_offset+i;
    out(2)=bot_cntrd;
%     
%     out(2)=cntrd(2)+cntrd_offset+i;
%     out(1)=bot_cntrd;  
%     
    cart=[cart;out];


    else
        
    end
    carte=sortrows(cart);
    x=carte(:,1);
    y=carte(:,2);  
    
end