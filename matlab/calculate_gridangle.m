function [STD_final, relangle]=calculate_gridangle(cntrd,I,cntrd_offset,searchradius,verbose,skip,skipstart,skiplength)
%% Auxiliary function to calculate the relative angle between the grid and the camera in the frame

%verbose=1;
%searchradius=300;
%cntrd_offset=400;

if skip==1
    skipstart=skipstart-cntrd_offset;
    loopvar=[1:skipstart-1 skipstart+skiplength:searchradius];
else
    loopvar=1:searchradius;
end

for i=loopvar
    
    top=I(floor(cntrd(1)-cntrd_offset-i),:);
    bot=I(floor(cntrd(1)+cntrd_offset+i),:);
    
    if (sum(top) ~= 0) && (sum(bot) ~= 0)    
    top_weights=repmat(1:size(top,2),1).*top;
    top_cntrd=sum(top_weights)./sum(top);
    
    bot_weights=repmat(1:size(bot,2),1).*bot;
    bot_cntrd=sum(bot_weights)./sum(bot);
    
    angle_top=atand((top_cntrd-cntrd(2))/(cntrd_offset+i));
    angle_bot=atand((bot_cntrd-cntrd(2))/(cntrd_offset+i));
    
    %angle_diff(i)=(angle_top+(angle_bot)/2);
    angle_diff(i)=(angle_top-(angle_bot));
    
    radius_vector(i)=cntrd_offset+i;
    
    for_STD_angle_diff=angle_diff;
    for_STD_angle_diff(angle_diff==0)=[];
    STD_vec(i)=std(for_STD_angle_diff);
       
    end

end

angle_diff_plot=angle_diff;
angle_diff_plot(angle_diff==0)=nan;

STD_vec_plot=STD_vec;
STD_vec_plot(STD_vec==0)=nan;

STD_vec_mean=STD_vec;
STD_vec_mean(STD_vec==0)=[];
STD_angdiff_mean=mean(STD_vec_mean);

angle_diff_mean=angle_diff;
angle_diff_mean(angle_diff==0)=[];
relangle=mean(angle_diff_mean);

STD_final=std(angle_diff_mean);

if verbose==1
    
    actual_radius=cntrd_offset+1:1:size(angle_diff_plot,2)+cntrd_offset;
    
    yyaxis left
    figure(2000),plot(actual_radius,angle_diff_plot),ylabel('Relative angle (°)'),xlabel('Distance from center (pix)');
    yline(relangle); ylim([-0.5+relangle relangle+0.5]);

    yyaxis right
    plot(actual_radius,STD_vec_plot), ylabel('Error (°)'), ylim([0 0.25])
    text(0.02,0.95,['Relative angle is: ',num2str(round(relangle,3)),'° ', char(177) ,num2str(round(STD_final,3)),'°'],'Units','Normalized')
    grid on, grid minor
    hold off 
    yyaxis left
    yline(relangle-0.05,'--k');
    yline(relangle+0.05,'--k');    
    legend('Relative angle','Mean relative angle','Accumulated Error (STD)','Location','NorthEast')

    %display(['Relative angle is: ',num2str(relangle),'°'])
end

end