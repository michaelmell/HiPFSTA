function void = drawOverlay(axes_handle,coordinates,color,linewidth)

hold on;
plot(coordinates(:,2), coordinates(:,1), color, 'Linewidth', linewidth,'Parent',axes_handle);
hold off;
